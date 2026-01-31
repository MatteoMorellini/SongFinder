import librosa
import numpy as np
import soundfile as sf
from .config import TARGET_SR, N_FFT, HOP_LENGTH
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path
import torch
import torchaudio

def load_audio(path):
    signal, sr = sf.read(path)
    return np.asarray(signal), sr

def cut_audio(signal, sample_rate, clip_length_sec):
    np.random.seed(42)
    total_samples = len(signal)
    clip_samples = int(clip_length_sec * sample_rate)
    start = np.random.randint(0, total_samples - clip_samples)
    end = start + clip_samples
    return signal[start:end]

def inject_noise(signal, snr_db):
    """
    Add white Gaussian noise to `signal` to get the desired SNR in dB.
    Assumes `signal` is a 1D float numpy array.
    """

    # make sure we work in float
    signal = signal.astype(float)

    # signal power (mean square)
    signal_power = np.mean(signal ** 2)

    if signal_power == 0:
        # silent signal, just return it (or raise)
        return signal

    # desired noise power
    noise_power = signal_power / (10 ** (snr_db / 10))

    # noise standard deviation
    noise_std = np.sqrt(noise_power)

    # generate white Gaussian noise
    noise = np.random.normal(0.0, noise_std, size=signal.shape)

    # noisy signal
    return signal + noise

def find_peaks(spectrogram, bands):
    peaks = []  # list of (time_index, freq_bin_index, amplitude)

    n_freq_bins, n_time_bins = spectrogram.shape
    # spectrogram has y-axis the values of the frequency bins: k in (0, 1, 2, ..., n_freq_bins-1) 
    # only to display they are converted to Hz: F(k) = k * sample_rate / n_fft

    for t in range(n_time_bins):
        for (f_lo, f_hi) in bands:
            
            # ensure we stay inside spectrogram
            f_hi = min(f_hi, n_freq_bins - 1)

            # slice magnitude values in this band
            band_slice = spectrogram[f_lo:f_hi+1, t]

            # find index of maximum inside the band
            local_idx = np.argmax(band_slice)
            global_idx = f_lo + local_idx  # convert back to absolute freq bin

            amp = band_slice[local_idx]
            peaks.append((t, global_idx, amp))
            
    return peaks

def plot_spectrogram_and_save(spectrogram, sample_rate, hop_length, peaks, freqs, output_path: Path):
    plot_peaks = peaks[::200] # plot only every 200th peak for visibility
    
    plot_times = [t for (t, fb, amp) in plot_peaks]
    plot_freqs = [freqs[fb] for (t, fb, amp) in plot_peaks]


    plot_times_sec = [t * hop_length / sample_rate for t in plot_times]

    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_spectrogram, sr=sample_rate, x_axis='time', y_axis='log', hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    # Overlay peaks (white dots)
    plt.scatter(plot_times_sec, plot_freqs,s=8, c='white', marker='o', alpha=0.8)
    plt.savefig(output_path)
    plt.close()

"""
Optimized spectrogram extraction for audio fingerprinting.

Multiple strategies to speed up STFT computation, which is often the bottleneck.
"""

import numpy as np
import librosa
import scipy.signal
from .config import N_FFT, TARGET_SR, HOP_LENGTH


# ============================================================================
# STRATEGY 1: Use scipy.signal.stft (often faster than librosa)
# ============================================================================

def extract_spectrogram_scipy(signal, sample_rate):
    """
    Use scipy's STFT implementation which is often faster than librosa.
    
    Expected speedup: 1.5-2x
    """
    if signal.ndim > 1:
        signal = librosa.to_mono(signal.T)
    
    signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=TARGET_SR)
    
    # scipy.signal.stft is often faster than librosa.stft
    f, t, stft = scipy.signal.stft(
        signal, 
        fs=TARGET_SR,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
        window='hann'
    )
    
    spectrogram = np.abs(stft)
    return spectrogram


# ============================================================================
# STRATEGY 2: Cache resampling (if processing same file multiple times)
# ============================================================================

def extract_spectrogram_cached_resample(signal, sample_rate, _cache={}):
    """
    Cache the resampled signal to avoid redundant resampling.
    Useful if you process the same audio multiple times.
    
    Note: Only use if you're repeatedly processing the same file!
    """
    if signal.ndim > 1:
        signal = librosa.to_mono(signal.T)
    
    # Create cache key from signal hash
    signal_hash = hash(signal.tobytes())
    cache_key = (signal_hash, sample_rate, TARGET_SR)
    
    if cache_key in _cache:
        signal = _cache[cache_key]
    else:
        signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=TARGET_SR)
        _cache[cache_key] = signal
    
    stft = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectrogram = np.abs(stft)
    return spectrogram


# ============================================================================
# STRATEGY 3: Skip resampling if already at target rate
# ============================================================================

def extract_spectrogram_smart_resample(signal, sample_rate):
    """
    Only resample if necessary. Many audio files are already at 44.1kHz or 22.05kHz.
    
    Expected speedup: 1.2-1.5x (depending on input sample rate)
    """
    if signal.ndim > 1:
        signal = librosa.to_mono(signal.T)
    
    # Only resample if needed
    if sample_rate != TARGET_SR:
        signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=TARGET_SR)
    
    stft = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectrogram = np.abs(stft)
    return spectrogram


# ============================================================================
# STRATEGY 4: Lower resolution STFT (trade accuracy for speed)
# ============================================================================

def extract_spectrogram_low_res(signal, sample_rate, n_fft_override=1024):
    """
    Use smaller FFT window for faster computation.
    
    WARNING: This may reduce fingerprint quality!
    Use only if you can tolerate some accuracy loss.
    
    n_fft=1024 instead of 2048 → ~2x faster STFT
    
    Expected speedup: 2-3x
    Accuracy impact: 5-10% lower recognition rate (estimate)
    """
    if signal.ndim > 1:
        signal = librosa.to_mono(signal.T)
    
    signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=TARGET_SR)
    
    # Adjust hop_length proportionally
    hop_length_adjusted = HOP_LENGTH * n_fft_override // N_FFT
    
    stft = librosa.stft(signal, n_fft=n_fft_override, hop_length=hop_length_adjusted)
    spectrogram = np.abs(stft)
    return spectrogram


# ============================================================================
# STRATEGY 5: Combined optimization (RECOMMENDED)
# ============================================================================

def extract_spectrogram_optimized(signal, sample_rate):
    """
    Combines multiple optimizations for best performance.
    
    - Uses scipy.signal.stft (faster)
    - Skips unnecessary resampling
    - Uses float32 instead of float64 (less memory, faster)
    
    Expected speedup: 2-3x
    """
    if signal.ndim > 1:
        signal = librosa.to_mono(signal.T)
    
    # Convert to float32 for faster processing
    signal = signal.astype(np.float32)
    
    # Only resample if needed
    if sample_rate != TARGET_SR:
        signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=TARGET_SR)
    
    # Use scipy's STFT (often faster)
    f, t, stft = scipy.signal.stft(
        signal,
        fs=TARGET_SR,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
        window='hann',
        boundary=None,  # Disable boundary extension for speed
        padded=False     # Disable zero-padding for speed
    )
    
    # Compute magnitude
    spectrogram = np.abs(stft).astype(np.float32)
    
    return spectrogram


# ============================================================================
# STRATEGY 6: Parallel processing (for batch indexing)
# ============================================================================

def extract_spectrogram_batch(signals_and_rates, n_jobs=-1):
    """
    Process multiple spectrograms in parallel using joblib.
    
    Useful when indexing many songs at once.
    
    Args:
        signals_and_rates: List of (signal, sample_rate) tuples
        n_jobs: Number of parallel jobs (-1 = use all cores)
    
    Returns:
        List of spectrograms
    """
    from joblib import Parallel, delayed
    
    def process_one(signal, sr):
        return extract_spectrogram_optimized(signal, sr)
    
    spectrograms = Parallel(n_jobs=n_jobs)(
        delayed(process_one)(sig, sr) for sig, sr in signals_and_rates
    )
    
    return spectrograms


# ============================================================================
# STRATEGY 7: GPU acceleration (if CUDA available)
# ============================================================================

def extract_spectrogram_gpu(signal, sample_rate):
    """
    Use GPU-accelerated STFT via CuPy (if available).
    
    Requires: pip install cupy-cuda11x (or cupy-cuda12x)
    
    Expected speedup: 5-10x on GPU
    """
    try:
        import cupy as cp
        import cupyx.scipy.signal
        
        if signal.ndim > 1:
            signal = librosa.to_mono(signal.T)
        
        if sample_rate != TARGET_SR:
            signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=TARGET_SR)
        
        # Move to GPU
        signal_gpu = cp.asarray(signal, dtype=cp.float32)
        
        # Compute STFT on GPU
        f, t, stft = cupyx.scipy.signal.stft(
            signal_gpu,
            fs=TARGET_SR,
            nperseg=N_FFT,
            noverlap=N_FFT - HOP_LENGTH,
            window='hann'
        )
        
        # Compute magnitude and move back to CPU
        spectrogram = cp.abs(stft).get().astype(np.float32)
        
        return spectrogram
        
    except ImportError:
        print("CuPy not available, falling back to CPU")
        return extract_spectrogram_optimized(signal, sample_rate)


# ============================================================================
# ORIGINAL (for comparison)
# ============================================================================

def extract_spectrogram_original(signal, sample_rate):
    """Original implementation for comparison."""
    if signal.ndim > 1:
        signal = librosa.to_mono(signal.T)
    
    signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=TARGET_SR)
    stft = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectrogram = np.abs(stft)
    
    return spectrogram


# ============================================================================
# Convenience function to choose best strategy
# ============================================================================

def extract_spectrogram(signal, sample_rate, strategy='optimized'):
    """
    Extract spectrogram with selectable optimization strategy.
    
    Args:
        signal: Audio signal array
        sample_rate: Sample rate in Hz
        strategy: One of:
            - 'original': Your original implementation
            - 'scipy': Use scipy.signal.stft
            - 'optimized': Combined optimizations (RECOMMENDED)
            - 'low_res': Faster but lower quality
            - 'gpu': GPU acceleration if available
    
    Returns:
        2D spectrogram array (freq_bins, time_frames)
    """
    strategies = {
        'original': extract_spectrogram_original,
        'scipy': extract_spectrogram_scipy,
        'optimized': extract_spectrogram_optimized,
        'low_res': extract_spectrogram_low_res,
        'gpu': extract_spectrogram_gpu,
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")
    
    return strategies[strategy](signal, sample_rate)

# ============================================================================
# Switch from librosa to torchaudio
# ============================================================================

def extract_spectrogram_fast(signal, sample_rate):
    # signal: np.ndarray float32/float64, shape (n,) or (n, ch)
    x = torch.tensor(signal)

    if x.ndim > 1:
        x = x.mean(dim=-1)  # mono (if shape is (n, ch); adapt if reversed)

    x = x.to(torch.float32)

    # resample
    x = torchaudio.functional.resample(x, orig_freq=sample_rate, new_freq=TARGET_SR)

    # stft
    window = torch.hann_window(N_FFT, device=x.device)
    stft = torch.stft(
        x, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window,
        return_complex=True, center=True, pad_mode="reflect"
    )
    spec = stft.abs().cpu().numpy()
    return spec
