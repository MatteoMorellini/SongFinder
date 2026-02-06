"""Audio augmentation utilities for benchmarking."""

import numpy as np
import random
import subprocess
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
from typing import List, Optional, Tuple
from scipy.signal import fftconvolve


def load_audio(audio_path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """Load and resample audio file using soundfile with ffmpeg fallback."""
    try:
        waveform, sr = sf.read(audio_path)
        waveform = np.atleast_1d(waveform).flatten() if waveform.ndim == 1 else waveform.mean(axis=1)
    except Exception:
        # Fallback to ffmpeg for ISO Media / ALAC
        try:
            probe = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                 '-show_entries', 'stream=sample_rate', '-of', 'csv=p=0', str(audio_path)],
                capture_output=True, text=True, timeout=10, check=True
            )
            sr = int(probe.stdout.strip())

            result = subprocess.run(
                ['ffmpeg', '-i', str(audio_path), '-f', 'f32le', '-acodec', 'pcm_f32le',
                 '-ac', '1', '-'],
                capture_output=True, timeout=60, check=True
            )
            waveform = np.frombuffer(result.stdout, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load {audio_path}: {e}")

    # Resample using torchaudio (3.6x faster than scipy)
    if sr != target_sr:
        waveform_torch = torch.from_numpy(waveform)
        waveform = torchaudio.functional.resample(
            waveform_torch, sr, target_sr
        ).numpy().astype(np.float32)

    return waveform.astype(np.float32), target_sr


def apply_ir(signal: np.ndarray, ir_path: Path, sample_rate: int,
             save_example_dir: Optional[Path] = None, example_prefix: str = "aug") -> np.ndarray:
    """Apply impulse response convolution (reverb) to the signal."""
    try:
        ir, _ = load_audio(ir_path, sample_rate)
        ir = ir / (np.max(np.abs(ir)) + 1e-10)

        signal_1d = np.atleast_1d(signal).flatten()

        convolved = fftconvolve(signal_1d, ir, mode='same')
        max_orig = np.max(np.abs(signal_1d))
        if max_orig > 0:
            convolved = convolved / (np.max(np.abs(convolved)) + 1e-10) * max_orig

        result = convolved.astype(np.float32)

        if save_example_dir is not None:
            try:
                existing = list(save_example_dir.glob(f"{example_prefix}_*.wav"))
                if len(existing) < 5:
                    out_path = save_example_dir / f"{example_prefix}_{len(existing)}_{ir_path.stem}.wav"
                    sf.write(out_path, result, sample_rate)
            except Exception:
                pass

        return result

    except Exception as e:
        print(f"Warning: Failed to apply IR from {ir_path}: {e}")
        return signal


def add_noise(signal: np.ndarray, snr_db: float, noise_files: List[Path], sample_rate: int) -> np.ndarray:
    """Add noise to signal at specified SNR."""
    signal_1d = np.atleast_1d(signal).flatten()

    if not noise_files:
        signal_power = np.mean(signal_1d ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal_1d))
        return (signal_1d + noise).astype(np.float32)

    noise_file = random.choice(noise_files)
    try:
        noise, _ = load_audio(noise_file, sample_rate)

        if len(noise) < len(signal_1d):
            noise = np.tile(noise, int(np.ceil(len(signal_1d) / len(noise))))
        noise = noise[:len(signal_1d)]

        signal_power = np.mean(signal_1d ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        noise = noise * np.sqrt(target_noise_power / noise_power)

        return (signal_1d + noise).astype(np.float32)

    except Exception as e:
        print(f"Warning: Failed to add noise from {noise_file}: {e}")
        return signal


def create_augmented_query(
    audio_path: Path,
    clip_length_sec: float,
    target_sr: int,
    snr_db: Optional[float] = None,
    use_ir: bool = False,
    ir_files: Optional[List[Path]] = None,
    noise_files: Optional[List[Path]] = None,
    save_example_dir: Optional[Path] = None,
    example_prefix: str = "query",
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Create an augmented query clip: load -> cut -> IR reverb -> noise.
    Deterministic when seed is provided (same seed = same augmentation).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    waveform, sr = load_audio(audio_path, target_sr)

    clip_samples = int(clip_length_sec * sr)
    if len(waveform) > clip_samples:
        start = random.randint(0, len(waveform) - clip_samples)
        waveform = waveform[start:start + clip_samples]

    if use_ir and ir_files:
        ir_file = random.choice(ir_files)
        waveform = apply_ir(waveform, ir_file, sr, save_example_dir, example_prefix)

    if snr_db is not None and noise_files:
        waveform = add_noise(waveform, snr_db, noise_files, sr)

    return waveform
