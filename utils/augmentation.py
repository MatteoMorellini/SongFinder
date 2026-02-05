"""
Centralized audio augmentation functions for benchmarking.
Ensures consistent audio processing for both Shazam and GraFP.
Uses torchaudio for fast resampling (3.6x faster than scipy).
"""

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
    """
    Load and resample audio file.
    Uses soundfile (fast) with ffmpeg fallback (robust).
    Uses torchaudio for resampling (3.6x faster than scipy).

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate

    Returns:
        Tuple of (audio signal as 1D numpy array, sample rate)
    """
    # Try soundfile first (fastest)
    try:
        waveform, sr = sf.read(audio_path)
        waveform = np.atleast_1d(waveform).flatten() if waveform.ndim == 1 else waveform.mean(axis=1)
    except Exception:
        # Fallback to ffmpeg for ISO Media/ALAC
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


def apply_ir(signal: np.ndarray, ir_path: Path, sample_rate: int, save_example_dir: Optional[Path] = None, example_prefix: str = "aug") -> np.ndarray:
    """
    Apply impulse response convolution (reverb) to the signal.
    
    Args:
        signal: Input audio signal (1D numpy array)
        ir_path: Path to the impulse response WAV file
        sample_rate: Sample rate of the input signal
        save_example_dir: Optional directory to save examples
        example_prefix: Prefix for saved example files
    
    Returns:
        Convolved signal with same length as input
    """
    try:
        # Load IR file using our robust loader
        ir, _ = load_audio(ir_path, sample_rate)
        
        # Normalize IR
        ir = ir / (np.max(np.abs(ir)) + 1e-10)
        
        # Ensure signal is 1D
        signal_1d = np.atleast_1d(signal).flatten()
        
        # Convolve and normalize to preserve original level
        convolved = fftconvolve(signal_1d, ir, mode='same')
        max_orig = np.max(np.abs(signal_1d))
        if max_orig > 0:
            convolved = convolved / (np.max(np.abs(convolved)) + 1e-10) * max_orig
        
        result = convolved.astype(np.float32)
        
        # Save example if requested
        if save_example_dir is not None:
            try:
                import soundfile as sf
                existing = list(save_example_dir.glob(f"{example_prefix}_*.wav"))
                if len(existing) < 5:
                    out_path = save_example_dir / f"{example_prefix}_{len(existing)}_{ir_path.stem}.wav"
                    sf.write(out_path, result, sample_rate)
            except:
                pass
        
        return result
        
    except Exception as e:
        print(f"Warning: Failed to apply IR from {ir_path}: {e}")
        return signal


def add_noise(signal: np.ndarray, snr_db: float, noise_files: List[Path], sample_rate: int) -> np.ndarray:
    """
    Add noise to signal at specified SNR.
    
    Args:
        signal: Input audio signal (1D numpy array)
        snr_db: Signal-to-noise ratio in dB
        noise_files: List of noise file paths
        sample_rate: Sample rate of the signal
    
    Returns:
        Noisy signal
    """
    signal_1d = np.atleast_1d(signal).flatten()
    
    if not noise_files:
        # Fallback to white noise
        signal_power = np.mean(signal_1d ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal_1d))
        return (signal_1d + noise).astype(np.float32)
    
    noise_file = random.choice(noise_files)
    try:
        # Load noise using our robust loader
        noise, _ = load_audio(noise_file, sample_rate)
        
        # Tile or truncate to match signal length
        if len(noise) < len(signal_1d):
            noise = np.tile(noise, int(np.ceil(len(signal_1d) / len(noise))))
        noise = noise[:len(signal_1d)]
        
        # Scale noise to desired SNR
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
    Create an augmented query audio with specified transformations.
    This is the single source of truth for audio augmentation in benchmarks.
    
    Args:
        audio_path: Path to the source audio file
        clip_length_sec: Length of clip to extract in seconds
        target_sr: Target sample rate
        snr_db: Optional SNR for noise injection (None = no noise)
        use_ir: Whether to apply impulse response
        ir_files: List of IR files to choose from
        noise_files: List of noise files to choose from
        save_example_dir: Optional directory to save examples
        example_prefix: Prefix for saved example files
        seed: Optional seed for reproducibility (same seed = same augmentation)
    
    Returns:
        Augmented audio signal as 1D numpy array
    """
    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Load and resample
    waveform, sr = load_audio(audio_path, target_sr)
    
    # Cut clip
    clip_samples = int(clip_length_sec * sr)
    if len(waveform) > clip_samples:
        start = random.randint(0, len(waveform) - clip_samples)
        waveform = waveform[start:start + clip_samples]
    
    # Apply IR (reverb) first
    if use_ir and ir_files:
        ir_file = random.choice(ir_files)
        waveform = apply_ir(waveform, ir_file, sr, save_example_dir, example_prefix)
    
    # Add noise second
    if snr_db is not None and noise_files:
        waveform = add_noise(waveform, snr_db, noise_files, sr)
    
    return waveform

