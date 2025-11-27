import librosa
import numpy as np
import soundfile as sf
from .config import TARGET_SR, N_FFT, HOP_LENGTH
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path

def load_audio(path):
    signal, sr = sf.read(path)
    return np.asarray(signal), sr

def extract_spectrogram(signal, sample_rate):
    if signal.ndim > 1:
        # signal is (num_samples, num_channels); transpose to (channels, samples)
        signal = librosa.to_mono(signal.T)  # now 1D
        
    # "naive" downsample to 11025 Hz for faster processing and noise reduction
    # equivalent to a f_max of ~5kHz
    signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=TARGET_SR)
    #hop_length = 512  # default when using n_fft=2048 with librosa.stft
    # Desire 30 fps, where #fps = sample_rate / hop_length => hop_length = sample_rate / 30 
    # Thus, hop_length = 11025 / 30 = 367.5 ~ 368
    stft = librosa.stft(signal, n_fft = N_FFT, hop_length=HOP_LENGTH)
    spectrogram = np.abs(stft) # since stft is a complex number, we take the magnitude (real part))
    return spectrogram

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