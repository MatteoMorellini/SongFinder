import soundfile as sf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa

def find_peaks(spectrogram, n_time_bins, n_freq_bins,bands):
    peaks = []  # list of (time_index, freq_bin_index, amplitude)

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

def plot_spectrogram_and_save(signal, sample_rate, output_path: Path, bands):
    if signal.ndim > 1:
        # signal is (num_samples, num_channels); transpose to (channels, samples)
        signal = librosa.to_mono(signal.T)  # now 1D
        
    # "naive" downsample to 11025 Hz for faster processing and noise reduction
    # equivalent to a f_max of ~5kHz
    signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=11025)
    sample_rate = 11025

    stft = librosa.stft(signal, n_fft = 2048)
    spectrogram = np.abs(stft) # since stft is a complex number, we take the magnitude (real part))

    n_freq_bins, n_time_bins = spectrogram.shape
    # spectrogram has y-axis the values of the frequency bins: k in (0, 1, 2, ..., n_freq_bins-1) 
    # only to display they are converted to Hz: F(k) = k * sample_rate / n_fft
    print("freq bins:", n_freq_bins, "time bins:", n_time_bins)

    print(len(spectrogram, n_time_bins, n_freq_bins, bands))
    return # ! 
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_spectrogram, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    plt.savefig(output_path)
    plt.close()

def main():
    signal, sample_rate = sf.read(Path('data') / 'osakablues.flac')
    print(f"Sample Rate: {sample_rate}")

    bands = [
        (0, 10),      # very low
        (10, 20),     # low
        (20, 40),     # low-mid
        (40, 80),     # mid
        (80, 160),    # mid-high
        (160, 511)    # high
    ]

    plot_spectrogram_and_save(signal, sample_rate, Path('imgs') / 'spectrogram.png', bands)




if __name__ == '__main__':
    main()