#!/usr/bin/env python3
"""
Benchmark script: compare Shazam vs GraFP inference performance.

Prerequisites:
    1. Run preprocess.py first to generate fingerprints
    2. Have audio files for query testing

Usage:
    python scripts/benchmark.py --db_dir ./fingerprints/ \
                                --test_dir ~/datasets/fma_small \
                                --aug_dir ~/datasets/aug \
                                --n_test 100
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torchaudio


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
    ir_files = list(ir_dir.rglob("*.wav"))
    return ir_files


def add_noise(signal: np.ndarray, snr_db: float, noise_files: List[Path], sr: int) -> np.ndarray:
    """Add noise to signal at specified SNR."""
    if not noise_files:
        # Fallback to white noise
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        return signal + noise
    
    # Load random noise file
    noise_file = random.choice(noise_files)
    try:
        noise, noise_sr = torchaudio.load(noise_file)
        noise = noise.mean(dim=0).numpy()
        
        if noise_sr != sr:
            # Simple resampling
            noise = np.interp(
                np.linspace(0, len(noise), int(len(noise) * sr / noise_sr)),
                np.arange(len(noise)),
                noise
            )
        
        # Tile or truncate to match signal length
        if len(noise) < len(signal):
            noise = np.tile(noise, int(np.ceil(len(signal) / len(noise))))
        noise = noise[:len(signal)]
        
        # Scale noise to desired SNR
        signal_power = np.mean(signal ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        noise = noise * np.sqrt(target_noise_power / noise_power)
        
        return signal + noise
    except:
        return signal


def apply_ir(signal: np.ndarray, ir_files: List[Path], sr: int) -> np.ndarray:
    """Apply impulse response convolution."""
    if not ir_files:
        return signal
    
    ir_file = random.choice(ir_files)
    try:
        from scipy.signal import fftconvolve
        
        ir, ir_sr = torchaudio.load(ir_file)
        ir = ir.mean(dim=0).numpy()
        
        if ir_sr != sr:
            ir = np.interp(
                np.linspace(0, len(ir), int(len(ir) * sr / ir_sr)),
                np.arange(len(ir)),
                ir
            )
        
        # Convolve and normalize
        convolved = fftconvolve(signal, ir, mode='same')
        convolved = convolved / (np.max(np.abs(convolved)) + 1e-10) * np.max(np.abs(signal))
        
        return convolved.astype(np.float32)
    except:
        return signal


def create_query(
    audio_path: Path,
    condition: TestCondition,
    noise_files: List[Path],
    ir_files: List[Path],
    target_sr: int = 8000
) -> np.ndarray:
    """Create a query audio with specified augmentation."""
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0).numpy()
    
    # Resample
    if sr != target_sr:
        waveform = np.interp(
            np.linspace(0, len(waveform), int(len(waveform) * target_sr / sr)),
            np.arange(len(waveform)),
            waveform
        ).astype(np.float32)
        sr = target_sr
    
    # Cut clip
    clip_samples = int(condition.clip_length_sec * sr)
    if len(waveform) > clip_samples:
        start = random.randint(0, len(waveform) - clip_samples)
        waveform = waveform[start:start + clip_samples]
    
    # Apply IR
    if condition.use_ir:
        waveform = apply_ir(waveform, ir_files, sr)
    
    # Add noise
    if condition.snr_db is not None:
        waveform = add_noise(waveform, condition.snr_db, noise_files, sr)
    
    return waveform


def benchmark_shazam(
    db_dir: Path,
    test_files: List[Path],
    noise_files: List[Path],
    ir_files: List[Path],
    conditions: List[TestCondition]
) -> BenchmarkResults:
    """Benchmark Shazam approach."""
    from approaches.shazam import ShazamRecognizer
    
    print("\n=== Shazam Benchmark ===")
    
    # Load database
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
    
    for condition in conditions:
        correct = 0
        total = 0
        query_times = []
        
        for test_file in test_files:
            expected = test_file.stem
            
            try:
                start = time.time()
                song, score, _ = recognizer.recognize(
                    test_file,
                    clip_length_sec=condition.clip_length_sec,
                    snr_db=condition.snr_db
                )
                query_time = (time.time() - start) * 1000
                query_times.append(query_time)
                
                if song == expected:
                    correct += 1
                total += 1
                
            except Exception as e:
                total += 1
        
        accuracy = correct / total * 100 if total > 0 else 0
        avg_time = np.mean(query_times) if query_times else 0
        
        results.conditions[condition.name] = {
            "accuracy": accuracy,
            "avg_query_time_ms": avg_time,
            "correct": correct,
            "total": total
        }
        
        print(f"  {condition.name}: {accuracy:.1f}% ({correct}/{total}), {avg_time:.1f}ms/query")
    
    return results


def benchmark_grafp(
    db_dir: Path,
    test_files: List[Path],
    noise_files: List[Path],
    ir_files: List[Path],
    conditions: List[TestCondition],
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda"
) -> BenchmarkResults:
    """Benchmark GraFP approach."""
    from approaches.grafp import load_config, load_model
    from approaches.grafp.modules.transformations import AudioTransform
    from approaches.grafp.inference import recognize
    
    print("\n=== GraFP Benchmark ===")
    
    # Load config and model
    cfg = load_config(config_path)
    model = load_model(cfg, checkpoint_path)
    transform = AudioTransform(cfg).to(device)
    
    # Load database
    start = time.time()
    from approaches.grafp.inference import load_fingerprints
    db_fp, db_meta = load_fingerprints(db_dir / "grafp")
    db_load_time = (time.time() - start) * 1000
    
    results = BenchmarkResults(
        approach="GraFP",
        n_db_songs=len(set(db_meta)),
        n_queries=len(test_files),
        db_load_time_ms=db_load_time
    )
    
    print(f"Loaded {db_fp.shape[0]} fingerprints ({results.n_db_songs} songs) in {db_load_time:.1f}ms")
    
    model.eval()
    
    for condition in conditions:
        correct = 0
        total = 0
        query_times = []
        
        for test_file in test_files:
            expected = test_file.stem
            
            try:
                # Create augmented query
                query_audio = create_query(
                    test_file, condition, noise_files, ir_files, cfg['fs']
                )
                
                # Measure inference time only
                start = time.time()
                
                waveform = torch.from_numpy(query_audio).float()
                segments = transform(waveform.unsqueeze(0).to(device))
                
                with torch.no_grad():
                    _, _, query_fp, _ = model(segments, segments)
                
                song, votes = recognize(query_fp.cpu().numpy(), db_fp, db_meta)
                
                query_time = (time.time() - start) * 1000
                query_times.append(query_time)
                
                if song == expected:
                    correct += 1
                total += 1
                
            except Exception as e:
                print(f"    Error {test_file.name}: {e}")
                total += 1
        
        accuracy = correct / total * 100 if total > 0 else 0
        avg_time = np.mean(query_times) if query_times else 0
        
        results.conditions[condition.name] = {
            "accuracy": accuracy,
            "avg_query_time_ms": avg_time,
            "correct": correct,
            "total": total
        }
        
        print(f"  {condition.name}: {accuracy:.1f}% ({correct}/{total}), {avg_time:.1f}ms/query")
    
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
            
            print(f"\n  {cond_name}:")
            print(f"    {'Accuracy':<20} {s_acc:>12.1f}% {g_acc:>12.1f}% {'Shazam' if s_acc > g_acc else 'GraFP':>15}")
            print(f"    {'Query Time (ms)':<20} {s_time:>12.1f} {g_time:>12.1f} {'Shazam' if s_time < g_time else 'GraFP':>15}")
    
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
    
    results = {}
    
    # Shazam
    if not args.grafp_only and (db_dir / "shazam").exists():
        shazam_results = benchmark_shazam(db_dir, test_files, noise_files, ir_files, TEST_CONDITIONS)
        results["shazam"] = asdict(shazam_results)
    
    # GraFP
    if not args.shazam_only and args.checkpoint and (db_dir / "grafp").exists():
        grafp_results = benchmark_grafp(
            db_dir, test_files, noise_files, ir_files, TEST_CONDITIONS,
            args.config, args.checkpoint, args.device
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
