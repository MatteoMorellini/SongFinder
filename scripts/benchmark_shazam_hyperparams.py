#!/usr/bin/env python3
"""
Shazam Hyperparameter Benchmark Script.

This script tests the Shazam approach with different hyperparameter configurations
using the same test conditions as benchmark.py.

Hyperparameters tested:
    - FUZ_FACTOR: absorb small variations in frequency/time
    - TARGET_SR: target sample rate
    - N_FFT: FFT window size
    - HOP_RATIO: hop_length = n_fft / hop_ratio (e.g., 4 means 75% overlap)
    - FAN_OUT: number of target peaks paired with each anchor

Usage:
    python scripts/benchmark_shazam_hyperparams.py --db_dir ./fingerprints/ \
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
import importlib
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Tuple
from itertools import product
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

# Reuse utilities from benchmark.py
from benchmark import (
    TestCondition,
    BenchmarkResults as BaseBenchmarkResults,
    TEST_CONDITIONS,
    load_noise_files,
    load_ir_files,
)


@dataclass
class HyperparamConfig:
    """Configuration for Shazam hyperparameters."""
    fuz_factor: int = 2
    target_sr: int = 11025
    n_fft: int = 2048
    hop_ratio: int = 6  # hop_length = n_fft / hop_ratio (6 ≈ default 368 for n_fft=2048)
    fan_out: int = 5
    
    @property
    def hop_length(self) -> int:
        """Compute hop_length from n_fft and hop_ratio."""
        return self.n_fft // self.hop_ratio
    
    def __str__(self) -> str:
        return f"fuz{self.fuz_factor}_sr{self.target_sr}_nfft{self.n_fft}_hop{self.hop_length}(1/{self.hop_ratio})_fan{self.fan_out}"
    
    def to_dict(self) -> dict:
        return {
            "fuz_factor": self.fuz_factor,
            "target_sr": self.target_sr,
            "n_fft": self.n_fft,
            "hop_ratio": self.hop_ratio,
            "hop_length": self.hop_length,  # computed value for reference
            "fan_out": self.fan_out
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "HyperparamConfig":
        """Reconstruct config from dict (ignores computed hop_length)."""
        return cls(
            fuz_factor=d["fuz_factor"],
            target_sr=d["target_sr"],
            n_fft=d["n_fft"],
            hop_ratio=d["hop_ratio"],
            fan_out=d["fan_out"],
        )


@dataclass
class HyperparamBenchmarkResults:
    approach: str
    config: dict
    n_db_songs: int
    n_queries: int
    db_load_time_ms: float
    indexing_time_ms: float
    n_fingerprints: int = 0  # total fingerprints in hash table
    fingerprints_per_song: float = 0.0  # average fingerprints per song
    hash_table_size_mb: float = 0.0  # approximate memory size
    conditions: Dict[str, dict] = field(default_factory=dict)


# Hyperparameter grid to search
# hop_ratio: 4 = 75% overlap, 6 ≈ 83% overlap, 8 = 87.5% overlap
DEFAULT_HYPERPARAM_GRID = {
    # "target_sr": [8000, 11025, 16000],
    # "n_fft": [1024, 2048, 4096],
    # "fan_out": [3, 5, 10],
    "target_sr": [11025],
    "n_fft": [2048],
    "fan_out": [5],
}


def set_shazam_env_vars(config: HyperparamConfig) -> None:
    """Set environment variables for Shazam configuration."""
    os.environ["SHAZAM_FUZ_FACTOR"] = str(config.fuz_factor)
    os.environ["SHAZAM_TARGET_SR"] = str(config.target_sr)
    os.environ["SHAZAM_N_FFT"] = str(config.n_fft)
    os.environ["SHAZAM_HOP_LENGTH"] = str(config.hop_length)
    os.environ["SHAZAM_FAN_OUT"] = str(config.fan_out)


def reload_shazam_modules() -> None:
    """Reload Shazam modules to pick up new environment variables."""
    # Need to reload modules in dependency order
    modules_to_reload = [
        "approaches.shazam.config",
        "approaches.shazam.audio",
        "approaches.shazam.hashing",
        "approaches.shazam.recognizer",
        "approaches.shazam",
    ]
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])


def index_songs_with_config(
    song_files: List[Path],
    config: HyperparamConfig,
    temp_db_dir: Path
) -> Tuple[float, int, int]:
    """
    Index songs with a specific hyperparameter configuration.
    
    Returns:
        Tuple of (indexing_time_ms, n_songs_indexed, n_fingerprints)
    """
    # Set environment variables
    set_shazam_env_vars(config)
    reload_shazam_modules()
    
    # Import fresh recognizer
    from approaches.shazam import ShazamRecognizer
    
    recognizer = ShazamRecognizer()
    
    start = time.time()
    for song_file in tqdm(song_files, desc=f"Indexing with {config}", leave=False):
        try:
            recognizer.index_song(song_file)
        except Exception as e:
            print(f"    Error indexing {song_file.name}: {e}")
    indexing_time = (time.time() - start) * 1000
    
    # Count total fingerprints
    n_fingerprints = sum(len(v) for v in recognizer.hash_table.values())
    
    # Save to temp directory
    temp_db_dir.mkdir(parents=True, exist_ok=True)
    recognizer.save(temp_db_dir)
    
    return indexing_time, recognizer.num_indexed_songs, n_fingerprints


def benchmark_shazam_with_config(
    config: HyperparamConfig,
    db_dir: Path,
    test_files: List[Path],
    noise_files: List[Path],
    ir_files: List[Path],
    conditions: List[TestCondition],
    indexing_time_ms: float = 0.0,
    n_fingerprints: int = 0,
    preloaded_recognizer=None  # Reuse already-loaded recognizer
) -> HyperparamBenchmarkResults:
    """Benchmark Shazam with a specific hyperparameter configuration.
    
    Note: We don't use benchmark_shazam from benchmark.py here to avoid
    loading the recognizer twice (which would double RAM usage).
    
    Args:
        preloaded_recognizer: If provided, reuse this recognizer instead of loading.
                              Useful when skip_indexing=True to avoid reloading.
    """
    # Set environment variables and reload modules
    set_shazam_env_vars(config)
    reload_shazam_modules()
    
    print(f"\n=== Shazam Benchmark: {config} ===")
    
    # Use preloaded recognizer or load fresh
    if preloaded_recognizer is not None:
        recognizer = preloaded_recognizer
        db_load_time = 0.0  # Already loaded
        print("  Using preloaded recognizer")
    else:
        from approaches.shazam import ShazamRecognizer
        start = time.time()
        recognizer = ShazamRecognizer()
        actual_db_dir = db_dir.parent if db_dir.name == "shazam" else db_dir
        recognizer.load(actual_db_dir / "shazam")
        db_load_time = (time.time() - start) * 1000
    
    # Get fingerprint stats
    if n_fingerprints == 0:
        n_fingerprints = sum(len(v) for v in recognizer.hash_table.values())
    
    n_songs = recognizer.num_indexed_songs
    fingerprints_per_song = n_fingerprints / n_songs if n_songs > 0 else 0
    hash_table_size_mb = (n_fingerprints * 24) / (1024 * 1024)
    
    print(f"  Loaded {n_songs} songs in {db_load_time:.1f}ms")
    print(f"  Fingerprints: {n_fingerprints:,} ({fingerprints_per_song:.0f}/song), ~{hash_table_size_mb:.1f}MB")
    
    # Run benchmarks for each condition (reusing the loaded recognizer)
    results_conditions = {}
    
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
        
        results_conditions[condition.name] = {
            "accuracy": accuracy,
            "avg_query_time_ms": avg_time,
            "correct": correct,
            "total": total
        }
        
        print(f"  {condition.name}: {accuracy:.1f}% ({correct}/{total}), {avg_time:.1f}ms/query")
    
    # Return results
    return HyperparamBenchmarkResults(
        approach="Shazam",
        config=config.to_dict(),
        n_db_songs=n_songs,
        n_queries=len(test_files),
        db_load_time_ms=db_load_time,
        indexing_time_ms=indexing_time_ms,
        n_fingerprints=n_fingerprints,
        fingerprints_per_song=fingerprints_per_song,
        hash_table_size_mb=hash_table_size_mb,
        conditions=results_conditions
    )


def generate_hyperparam_configs(
    grid: Dict[str, List],
    vary_one_at_a_time: bool = True
) -> List[HyperparamConfig]:
    """
    Generate hyperparameter configurations to test.
    
    Args:
        grid: Dictionary mapping parameter names to lists of values
        vary_one_at_a_time: If True, only vary one parameter at a time from defaults.
                           If False, generate full grid search.
    
    Returns:
        List of HyperparamConfig objects
    """
    # Default configuration
    default = HyperparamConfig()
    configs = [default]  # Always include default
    
    if vary_one_at_a_time:
        # Vary one parameter at a time
        for param, values in grid.items():
            for value in values:
                if value != getattr(default, param):
                    new_config = HyperparamConfig(
                        fuz_factor=value if param == "fuz_factor" else default.fuz_factor,
                        target_sr=value if param == "target_sr" else default.target_sr,
                        n_fft=value if param == "n_fft" else default.n_fft,
                        hop_ratio=value if param == "hop_ratio" else default.hop_ratio,
                        fan_out=value if param == "fan_out" else default.fan_out,
                    )
                    configs.append(new_config)
    else:
        # Full grid search
        keys = list(grid.keys())
        for values in product(*[grid[k] for k in keys]):
            config_dict = dict(zip(keys, values))
            configs.append(HyperparamConfig(**config_dict))
    
    # Remove duplicates
    seen = set()
    unique_configs = []
    for c in configs:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
    
    return unique_configs


def print_summary(all_results: List[HyperparamBenchmarkResults]):
    """Print summary comparison of all configurations."""
    print("\n" + "=" * 120)
    print("HYPERPARAMETER BENCHMARK SUMMARY")
    print("=" * 120)
    
    # Accuracy & Query Time Table
    print(f"\n{'Configuration':<45} {'clean_10s':>9} {'clean_5s':>9} {'snr_5db':>9} {'Avg Query':>11} {'FP/song':>10} {'Size MB':>9}")
    print("-" * 120)
    
    for result in all_results:
        config_str = str(HyperparamConfig.from_dict(result.config))[:43]
        clean_10s = result.conditions.get("clean_10s", {}).get("accuracy", 0)
        clean_5s = result.conditions.get("clean_5s", {}).get("accuracy", 0)
        snr_5db = result.conditions.get("snr_5db", {}).get("accuracy", 0)
        
        avg_query_time = np.mean([
            c.get("avg_query_time_ms", 0) 
            for c in result.conditions.values()
        ])
        
        print(f"{config_str:<45} {clean_10s:>8.1f}% {clean_5s:>8.1f}% {snr_5db:>8.1f}% {avg_query_time:>9.1f}ms {result.fingerprints_per_song:>10.0f} {result.hash_table_size_mb:>8.1f}")
    
    print("=" * 120)
    
    # Find best configuration for each metric
    print("\nBest configurations:")
    
    # Best by accuracy for each condition
    print("\n  By accuracy:")
    for cond_name in TEST_CONDITIONS:
        best_result = None
        best_acc = -1
        for result in all_results:
            acc = result.conditions.get(cond_name.name, {}).get("accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                best_result = result
        
        if best_result:
            config_str = str(HyperparamConfig.from_dict(best_result.config))
            print(f"    {cond_name.name:<20}: {best_acc:>6.1f}% - {config_str}")
    
    # Best by query time
    print("\n  By query time (fastest):")
    best_result = min(all_results, key=lambda r: np.mean([c.get("avg_query_time_ms", float('inf')) for c in r.conditions.values()]))
    avg_time = np.mean([c.get("avg_query_time_ms", 0) for c in best_result.conditions.values()])
    print(f"    {avg_time:.1f}ms - {HyperparamConfig.from_dict(best_result.config)}")
    
    # Best by storage efficiency (accuracy / fingerprints)
    print("\n  By efficiency (accuracy per 1000 fingerprints):")
    def efficiency(r):
        acc = r.conditions.get("clean_10s", {}).get("accuracy", 0)
        return acc / (r.fingerprints_per_song / 1000) if r.fingerprints_per_song > 0 else 0
    best_result = max(all_results, key=efficiency)
    eff = efficiency(best_result)
    print(f"    {eff:.2f} acc/%K - {HyperparamConfig.from_dict(best_result.config)}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Shazam with different hyperparameters')
    parser.add_argument('--db_dir', type=str, default='./fingerprints',
                        help='Directory with preprocessed fingerprints')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory with test audio files')
    parser.add_argument('--aug_dir', type=str, default='~/datasets/aug',
                        help='Directory with noise/IR augmentation files')
    parser.add_argument('--n_test', type=int, default=100,
                        help='Number of test queries')
    parser.add_argument('--output', type=str, default='benchmark_shazam_hyperparams.json')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--full_grid', action='store_true',
                        help='Run full grid search instead of varying one param at a time')
    parser.add_argument('--conditions', type=str, nargs='+', default=None,
                        help='Specific conditions to test (default: all)')
    
    # Individual hyperparameter overrides
    parser.add_argument('--fuz_factors', type=int, nargs='+', default=None,
                        help='FUZ_FACTOR values to test')
    parser.add_argument('--target_srs', type=int, nargs='+', default=None,
                        help='TARGET_SR values to test')
    parser.add_argument('--n_ffts', type=int, nargs='+', default=None,
                        help='N_FFT values to test')
    parser.add_argument('--hop_ratios', type=int, nargs='+', default=None,
                        help='HOP_RATIO values to test (hop_length = n_fft / ratio)')
    parser.add_argument('--fan_outs', type=int, nargs='+', default=None,
                        help='FAN_OUT values to test')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    test_dir = Path(args.test_dir).expanduser()
    aug_dir = Path(args.aug_dir).expanduser()
    
    print(f"Test: {test_dir}")
    print(f"Aug: {aug_dir}")
    
    # Build hyperparameter grid
    grid = DEFAULT_HYPERPARAM_GRID.copy()
    if args.fuz_factors:
        grid["fuz_factor"] = args.fuz_factors
    if args.target_srs:
        grid["target_sr"] = args.target_srs
    if args.n_ffts:
        grid["n_fft"] = args.n_ffts
    if args.hop_ratios:
        grid["hop_ratio"] = args.hop_ratios
    if args.fan_outs:
        grid["fan_out"] = args.fan_outs
    
    # Generate configurations
    configs = generate_hyperparam_configs(grid, vary_one_at_a_time=not args.full_grid)
    print(f"\nTesting {len(configs)} hyperparameter configurations:")
    for i, cfg in enumerate(configs, 1):
        print(f"  {i}. {cfg}")
    
    # Select test conditions
    if args.conditions:
        conditions = [c for c in TEST_CONDITIONS if c.name in args.conditions]
    else:
        conditions = TEST_CONDITIONS
    print(f"\nTest conditions: {[c.name for c in conditions]}")
    
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
    
    all_results = []
    
    # Always load fingerprints once at the beginning (they're the same for all configs)
    db_dir = Path(args.db_dir).expanduser()
    shazam_db_path = db_dir / "shazam"
    if not shazam_db_path.exists():
        print(f"Error: {shazam_db_path} does not exist. Please run indexing first.")
        sys.exit(1)
    
    print(f"\nLoading fingerprints from {shazam_db_path}...")
    from approaches.shazam import ShazamRecognizer
    
    start = time.time()
    preloaded_recognizer = ShazamRecognizer()
    preloaded_recognizer.load(shazam_db_path)
    load_time = (time.time() - start) * 1000
    
    preloaded_n_fingerprints = sum(len(v) for v in preloaded_recognizer.hash_table.values())
    print(f"  Loaded {preloaded_recognizer.num_indexed_songs} songs ({preloaded_n_fingerprints:,} fingerprints) in {load_time:.1f}ms")
    print("  Note: Fingerprints are loaded once and reused for all hyperparameter configurations.")
    
    for config in configs:
        n_fingerprints = preloaded_n_fingerprints
        indexing_time = 0.0
        
        # Benchmark this configuration
        try:
            results = benchmark_shazam_with_config(
                config=config,
                db_dir=db_dir,
                test_files=test_files,
                noise_files=noise_files,
                ir_files=ir_files,
                conditions=conditions,
                indexing_time_ms=indexing_time,
                n_fingerprints=n_fingerprints,
                preloaded_recognizer=preloaded_recognizer
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error benchmarking config {config}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if all_results:
        print_summary(all_results)
    
    # Save results
    output_data = {
        "configs_tested": len(configs),
        "test_conditions": [c.name for c in conditions],
        "n_test_files": len(test_files),
        "results": [asdict(r) for r in all_results]
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
