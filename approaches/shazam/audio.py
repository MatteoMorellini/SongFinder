import os
import torch
import torchaudio
import numpy as np
import soundfile as sf

from .config import HOP_LENGTH


def load_audio(path):
    """Load audio file with fallback for non-standard formats (ISO Media/ALAC)."""
    import subprocess

    try:
        signal, sr = sf.read(path)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        return signal.astype(np.float32), sr
    except Exception:
        pass

    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
             '-show_entries', 'stream=sample_rate', '-of', 'csv=p=0', str(path)],
            capture_output=True, text=True, timeout=10, check=True
        )
        sr = int(probe.stdout.strip())

        result = subprocess.run(
            ['ffmpeg', '-i', str(path), '-f', 'f32le', '-acodec', 'pcm_f32le',
             '-ac', '1', '-'],
            capture_output=True, timeout=60, check=True
        )

        signal = np.frombuffer(result.stdout, dtype=np.float32)
        return signal, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")


def cut_audio(signal, sample_rate, clip_length_sec):
    np.random.seed(42)
    total_samples = len(signal)
    clip_samples = int(clip_length_sec * sample_rate)
    start = np.random.randint(0, total_samples - clip_samples)
    end = start + clip_samples
    return signal[start:end]


def find_peaks(spectrogram, bands):
    """Find spectral peaks per time frame: one peak (max amplitude) per frequency band."""
    peaks = []
    n_freq_bins, n_time_bins = spectrogram.shape

    for t in range(n_time_bins):
        for (f_lo, f_hi) in bands:
            f_hi = min(f_hi, n_freq_bins - 1)
            band_slice = spectrogram[f_lo:f_hi+1, t]
            local_idx = np.argmax(band_slice)
            global_idx = f_lo + local_idx
            amp = band_slice[local_idx]
            peaks.append((t, global_idx, amp))

    return peaks


def extract_spectrogram(signal, sample_rate):
    """Extract magnitude spectrogram using torchaudio for resampling and torch for STFT."""
    x = torch.tensor(signal)

    from .config import N_FFT as DEFAULT_N_FFT, TARGET_SR as DEFAULT_TARGET_SR
    n_fft = int(os.environ.get("SHAZAM_N_FFT", DEFAULT_N_FFT))
    target_sr = float(os.environ.get("SHAZAM_TARGET_SR", DEFAULT_TARGET_SR))

    if x.ndim > 1:
        x = x.mean(dim=-1)

    x = x.to(torch.float32)
    x = torchaudio.functional.resample(x, orig_freq=sample_rate, new_freq=target_sr)

    window = torch.hann_window(n_fft, device=x.device)
    stft = torch.stft(
        x, n_fft=n_fft, hop_length=HOP_LENGTH, window=window,
        return_complex=True, center=True, pad_mode="reflect"
    )
    spec = stft.abs().cpu().numpy()
    return spec
