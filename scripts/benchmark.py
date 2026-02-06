#!/usr/bin/env python3
"""
Benchmark script: compare Shazam vs GraFP inference performance.

Usage:
    python scripts/benchmark.py --db_dir ./fingerprints/ \
                                --test_dir ~/datasets/fma_small \
                                --aug_dir ~/datasets/aug \
                                --n_test 100
"""

import sys
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from utils.augmentation import create_augmented_query, load_audio


@dataclass
class TestCondition:
    name: str
    clip_length_sec: float
    snr_db: Optional[float] = None
    use_ir: bool = False


@dataclass
class BenchmarkResults:
    approach: str
    n_db_songs: int
    n_queries: int
    db_load_time_ms: float
    conditions: Dict[str, dict] = field(default_factory=dict)
    timings: Dict[str, Dict[str, float]] = field(default_factory=dict)  # Average timings per condition


# Test conditions to evaluate
TEST_CONDITIONS = [
    TestCondition("clean_10s", clip_length_sec=10.0),
    TestCondition("clean_5s", clip_length_sec=5.0),
    TestCondition("clean_3s", clip_length_sec=3.0),
    TestCondition("snr_10db", clip_length_sec=10.0, snr_db=10.0),
    TestCondition("snr_5db", clip_length_sec=10.0, snr_db=5.0),
    TestCondition("snr_0db", clip_length_sec=10.0, snr_db=0.0),
    TestCondition("ir_10s", clip_length_sec=10.0, use_ir=True),
    TestCondition("ir_snr_5db", clip_length_sec=10.0, snr_db=5.0, use_ir=True),
]


def load_noise_files(aug_dir: Path) -> List[Path]:
    """Load noise files from augmentation directory."""
    noise_dir = aug_dir / "noise" if (aug_dir / "noise").exists() else aug_dir
    noise_files = list(noise_dir.rglob("*.wav"))
    if not noise_files:
        noise_files = list(noise_dir.rglob("*.mp3"))
    return noise_files


def load_ir_files(aug_dir: Path) -> List[Path]:
    """Load impulse response files from augmentation directory."""
    ir_dir = aug_dir / "ir" if (aug_dir / "ir").exists() else aug_dir / "rir"
    if not ir_dir.exists():
        ir_dir = aug_dir
    return list(ir_dir.rglob("*.wav"))


def benchmark_shazam(
    db_dir: Path,
    test_files: List[Path],
    noise_files: List[Path],
    ir_files: List[Path],
    conditions: List[TestCondition],
    target_sr: int = 16000,
    example_dir: Optional[Path] = None
) -> BenchmarkResults:
    """Benchmark Shazam with on-the-fly augmentation using deterministic seeds."""
    from approaches.shazam import ShazamRecognizer

    print("\n=== Shazam Benchmark ===")
    print(f"Target Sample Rate: {target_sr} Hz")

    start = time.time()
    recognizer = ShazamRecognizer()
    recognizer.load(db_dir / "shazam")
    db_load_time = (time.time() - start) * 1000

    results = BenchmarkResults(
        approach="Shazam",
        n_db_songs=recognizer.num_indexed_songs,
        n_queries=len(test_files),
        db_load_time_ms=db_load_time
    )

    print(f"Loaded {results.n_db_songs} songs in {db_load_time:.1f}ms")

    for cond_idx, condition in enumerate(conditions):
        correct = 0
        total = 0
        query_times = []
        all_scores = []  # Track all confidence scores for average
        errors = []  # Track errors: {expected, predicted, score}
        timings_per_step = {}  # Will be populated dynamically

        for file_idx, test_file in enumerate(test_files):
            expected = test_file.stem
            # Deterministic seed based on condition and file index
            seed = cond_idx * 10000 + file_idx

            try:
                t_start_total = time.perf_counter()

                # Load and augment audio (like GraFP)
                t0 = time.perf_counter()
                query_audio = create_augmented_query(
                    audio_path=test_file,
                    clip_length_sec=condition.clip_length_sec,
                    target_sr=target_sr,
                    snr_db=condition.snr_db,
                    use_ir=condition.use_ir,
                    ir_files=ir_files if condition.use_ir else None,
                    noise_files=noise_files if condition.snr_db is not None else None,
                    seed=seed,
                    save_example_dir=example_dir,
                    example_prefix=f"shazam_{condition.name}"
                )
                t_load = time.perf_counter() - t0

                # Recognition (includes all internal steps)
                song, score, metadata = recognizer.recognize(
                    signal=query_audio,
                    sample_rate=target_sr,
                    debug=False
                )

                t_total = time.perf_counter() - t_start_total

                # Collect all timings (load_audio + internal recognizer timings)
                combined_timings = {"load_audio": t_load}
                if 'timings' in metadata:
                    combined_timings.update(metadata['timings'])
                combined_timings["total"] = t_total

                # Store timings per step
                for key, value in combined_timings.items():
                    if key not in timings_per_step:
                        timings_per_step[key] = []
                    timings_per_step[key].append(value)

                query_times.append(t_total * 1000)  # Convert to ms
                all_scores.append(float(score))  # Track confidence score

                if song == expected:
                    correct += 1
                else:
                    # Record error
                    errors.append({
                        "file": test_file.name,
                        "expected": expected,
                        "predicted": song if song else "None",
                        "score": float(score)
                    })
                total += 1

            except Exception as e:
                print(f"  Error {test_file.name}: {e}")
                errors.append({
                    "file": test_file.name,
                    "expected": expected,
                    "predicted": "ERROR",
                    "error_message": str(e)
                })
                total += 1

        accuracy = correct / total * 100 if total > 0 else 0
        avg_time = np.mean(query_times) if query_times else 0
        avg_confidence = np.mean(all_scores) if all_scores else 0

        # Calculate average timings per step
        avg_timings = {}
        for key in timings_per_step:
            values = timings_per_step[key]
            avg_timings[key] = float(np.mean(values))

        results.conditions[condition.name] = {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "avg_query_time_ms": avg_time,
            "correct": correct,
            "total": total,
            "errors": errors,  # List of all misclassifications
            "avg_timings_per_step": avg_timings  # Average time per processing step
        }

        if condition.name not in results.timings:
            results.timings[condition.name] = avg_timings

        print(f"  {condition.name}: {accuracy:.1f}% ({correct}/{total}), conf={avg_confidence:.3f}, {avg_time:.1f}ms/query")
        if errors:
            print(f"    Errors: {len(errors)}")

    return results


def benchmark_grafp(
    db_dir: Path,
    test_files: List[Path],
    noise_files: List[Path],
    ir_files: List[Path],
    conditions: List[TestCondition],
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    example_dir: Optional[Path] = None
) -> BenchmarkResults:
    """Benchmark GraFP with on-the-fly augmentation using deterministic seeds."""
    from approaches.grafp import load_config, load_model
    from approaches.grafp.modules.transformations import AudioTransform
    from approaches.grafp.inference import recognize

    print("\n=== GraFP Benchmark ===")

    cfg = load_config(config_path)
    model = load_model(cfg, checkpoint_path)
    transform = AudioTransform(cfg).to(device)
    target_sr = cfg['fs']  # Use GraFP's native sample rate

    print(f"Target Sample Rate: {target_sr} Hz")

    start = time.time()
    from approaches.grafp.inference import load_fingerprints, get_or_build_index
    db_fp, db_meta, db_metadata_table = load_fingerprints(db_dir / "grafp")
    db_load_time = (time.time() - start) * 1000

    index_path = db_dir / "grafp" / "index_ivfpq.faiss"
    index, was_loaded = get_or_build_index(db_fp, str(index_path), use_gpu=True)

    results = BenchmarkResults(
        approach="GraFP",
        n_db_songs=len(set(db_meta)),
        n_queries=len(test_files),
        db_load_time_ms=db_load_time
    )

    print(f"Loaded {db_fp.shape[0]} fingerprints ({results.n_db_songs} songs) in {db_load_time:.1f}ms")

    model.eval()

    for cond_idx, condition in enumerate(conditions):
        correct = 0
        total = 0
        query_times = []
        all_scores = []  # Track all confidence scores for average
        errors = []  # Track errors: {expected, predicted, score}
        timings_per_step = {
            "load_audio": [],
            "transform": [],
            "model_inference": [],
            "search": [],
            "total": []
        }

        for file_idx, test_file in enumerate(test_files):
            expected = test_file.stem
            # Same deterministic seed as Shazam for identical augmentation choices
            seed = cond_idx * 10000 + file_idx

            try:
                t_start_total = time.perf_counter()

                # Load and augment audio
                t0 = time.perf_counter()
                query_audio = create_augmented_query(
                    audio_path=test_file,
                    clip_length_sec=condition.clip_length_sec,
                    target_sr=target_sr,
                    snr_db=condition.snr_db,
                    use_ir=condition.use_ir,
                    ir_files=ir_files if condition.use_ir else None,
                    noise_files=noise_files if condition.snr_db is not None else None,
                    seed=seed,
                    save_example_dir=example_dir,
                    example_prefix=f"grafp_{condition.name}"
                )
                t_load = time.perf_counter() - t0

                # Transform audio to spectrogram segments
                t0 = time.perf_counter()
                waveform = torch.from_numpy(query_audio).float()
                segments = transform(waveform.unsqueeze(0).to(device))
                t_transform = time.perf_counter() - t0

                # Model inference
                t0 = time.perf_counter()
                with torch.no_grad():
                    _, _, query_fp, _ = model(segments, segments)
                t_inference = time.perf_counter() - t0

                # Search in database
                t0 = time.perf_counter()
                song, score = recognize(query_fp.cpu().numpy(), db_fp, db_meta, index)
                t_search = time.perf_counter() - t0

                t_total = time.perf_counter() - t_start_total

                # Store timings
                timings_per_step["load_audio"].append(t_load)
                timings_per_step["transform"].append(t_transform)
                timings_per_step["model_inference"].append(t_inference)
                timings_per_step["search"].append(t_search)
                timings_per_step["total"].append(t_total)

                query_times.append(t_total * 1000)  # Convert to ms
                all_scores.append(float(score))  # Track confidence score

                if song == expected:
                    correct += 1
                else:
                    # Record error
                    errors.append({
                        "file": test_file.name,
                        "expected": expected,
                        "predicted": song if song else "None",
                        "score": float(score)
                    })
                total += 1

            except Exception as e:
                print(f"    Error {test_file.name}: {e}")
                import traceback
                traceback.print_exc()
                errors.append({
                    "file": test_file.name,
                    "expected": expected,
                    "predicted": "ERROR",
                    "error_message": str(e)
                })
                total += 1

        accuracy = correct / total * 100 if total > 0 else 0
        avg_time = np.mean(query_times) if query_times else 0
        avg_confidence = np.mean(all_scores) if all_scores else 0

        # Calculate average timings per step
        avg_timings = {}
        for step, times in timings_per_step.items():
            if times:
                avg_timings[step] = float(np.mean(times))

        results.conditions[condition.name] = {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "avg_query_time_ms": avg_time,
            "correct": correct,
            "total": total,
            "errors": errors,  # List of all misclassifications
            "avg_timings_per_step": avg_timings  # Average time per processing step
        }

        if condition.name not in results.timings:
            results.timings[condition.name] = avg_timings

        print(f"  {condition.name}: {accuracy:.1f}% ({correct}/{total}), conf={avg_confidence:.3f}, {avg_time:.1f}ms/query")
        if errors:
            print(f"    Errors: {len(errors)}")

    return results


def print_comparison(shazam: BenchmarkResults, grafp: BenchmarkResults):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON: Shazam vs GraFP")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'Shazam':>15} {'GraFP':>15} {'Winner':>15}")
    print("-" * 80)

    print(f"{'DB Songs':<25} {shazam.n_db_songs:>15} {grafp.n_db_songs:>15} {'-':>15}")
    print(f"{'DB Load Time (ms)':<25} {shazam.db_load_time_ms:>15.1f} {grafp.db_load_time_ms:>15.1f} "
          f"{'Shazam' if shazam.db_load_time_ms < grafp.db_load_time_ms else 'GraFP':>15}")

    print("\nAccuracy by condition:")
    for cond_name in shazam.conditions:
        if cond_name in grafp.conditions:
            s_acc = shazam.conditions[cond_name]["accuracy"]
            g_acc = grafp.conditions[cond_name]["accuracy"]
            s_time = shazam.conditions[cond_name]["avg_query_time_ms"]
            g_time = grafp.conditions[cond_name]["avg_query_time_ms"]
            s_errors = len(shazam.conditions[cond_name].get("errors", []))
            g_errors = len(grafp.conditions[cond_name].get("errors", []))

            print(f"\n  {cond_name}:")
            print(f"    {'Accuracy':<20} {s_acc:>12.1f}% {g_acc:>12.1f}% {'Shazam' if s_acc > g_acc else 'GraFP':>15}")
            print(f"    {'Query Time (ms)':<20} {s_time:>12.1f} {g_time:>12.1f} {'Shazam' if s_time < g_time else 'GraFP':>15}")
            print(f"    {'Errors':<20} {s_errors:>12} {g_errors:>12} {'-':>15}")

    # Print detailed timing breakdown
    print("\n" + "=" * 80)
    print("DETAILED TIMING BREAKDOWN (average per query)")
    print("=" * 80)

    for cond_name in shazam.conditions:
        if cond_name in grafp.conditions and cond_name in shazam.timings and cond_name in grafp.timings:
            print(f"\n  {cond_name}:")
            s_timings = shazam.timings[cond_name]
            g_timings = grafp.timings[cond_name]

            # Get all unique timing keys
            all_keys = set(s_timings.keys()) | set(g_timings.keys())

            print(f"    {'Step':<30} {'Shazam (s)':>15} {'GraFP (s)':>15}")
            print(f"    {'-'*60}")
            for key in sorted(all_keys):
                s_val = s_timings.get(key, 0)
                g_val = g_timings.get(key, 0)
                print(f"    {key:<30} {s_val:>15.4f} {g_val:>15.4f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Benchmark Shazam vs GraFP')
    parser.add_argument('--db_dir', type=str, default='./fingerprints',
                        help='Directory with preprocessed fingerprints')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory with test audio files')
    parser.add_argument('--aug_dir', type=str, default='~/datasets/aug',
                        help='Directory with noise/IR augmentation files')
    parser.add_argument('--n_test', type=int, default=50,
                        help='Number of test queries')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='GraFP checkpoint (required for GraFP)')
    parser.add_argument('--config', type=str, default='approaches/grafp/config/grafp.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='benchmark_results.json')
    parser.add_argument('--shazam_only', action='store_true')
    parser.add_argument('--grafp_only', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    db_dir = Path(args.db_dir).expanduser()
    test_dir = Path(args.test_dir).expanduser()
    aug_dir = Path(args.aug_dir).expanduser()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"DB: {db_dir}")
    print(f"Test: {test_dir}")
    print(f"Aug: {aug_dir}")
    print(f"Device: {args.device}")
    
    # Load test files
    test_files = list(test_dir.rglob("*.mp3"))
    if not test_files:
        test_files = list(test_dir.rglob("*.flac"))
    
    random.shuffle(test_files)
    test_files = test_files[:args.n_test]
    print(f"Test files: {len(test_files)}")
    
    # Load augmentation files
    noise_files = load_noise_files(aug_dir)
    ir_files = load_ir_files(aug_dir)
    print(f"Noise files: {len(noise_files)}, IR files: {len(ir_files)}")
    
    # Create examples directory
    example_dir = Path("examples/reverb")
    example_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}

    # Shazam (uses 16000 Hz to match GraFP)
    if not args.grafp_only and (db_dir / "shazam").exists():
        shazam_results = benchmark_shazam(
            db_dir, test_files, noise_files, ir_files, TEST_CONDITIONS,
            target_sr=16000, example_dir=example_dir
        )
        results["shazam"] = asdict(shazam_results)
    
    # GraFP (uses its own sample rate from config)
    if not args.shazam_only and args.checkpoint and (db_dir / "grafp").exists():
        grafp_results = benchmark_grafp(
            db_dir, test_files, noise_files, ir_files, TEST_CONDITIONS,
            args.config, args.checkpoint, args.device, example_dir=example_dir
        )
        results["grafp"] = asdict(grafp_results)
    
    # Comparison
    if "shazam" in results and "grafp" in results:
        print_comparison(
            BenchmarkResults(**{k: v for k, v in results["shazam"].items() if k != 'conditions'}, 
                           conditions=results["shazam"]["conditions"]),
            BenchmarkResults(**{k: v for k, v in results["grafp"].items() if k != 'conditions'},
                           conditions=results["grafp"]["conditions"])
        )
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
