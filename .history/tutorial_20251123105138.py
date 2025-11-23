import soundfile as sf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa

def build_hashes(
    peaks,
    freqs,
    fan_out=5,
    dt_min_frames=1,      # ≥ 1 frame ahead
    dt_max_frames=30*6,     # ≤ ~1s ahead if ~30 fps
    freq_band_hz=None     # e.g. 1200 → ±1200 Hz around anchor; None = no limit
):
    """
    peaks:      list of (t_frame, f_bin, amplitude), assumed sorted by t_frame
    freqs:      1D array mapping frequency-bin index → Hz (librosa.fft_frequencies)
    sample_rate, hop_length: used only for reference / debugging
    fan_out:    max number of target points per anchor # ! interesting hyperparameter to twist in experiments
    returns:    list of hashes and metadata
                hashes: list of (f_anchor_bin, f_target_bin, dt_frames)
    """
    hashes = []

    n_peaks = len(peaks)

    for i in range(n_peaks):
        t_a, f_a, amp_a = peaks[i]

        # Collect candidates in the target zone ahead of this anchor
        candidates = []

        j = i + 1
        while j < n_peaks:
            t_b, f_b, amp_b = peaks[j]
            dt = t_b - t_a

            # stop when we're beyond the target window in time
            if dt > dt_max_frames:
                break

            if dt >= dt_min_frames:
                if freq_band_hz is not None:
                    # check frequency distance in Hz
                    if abs(freqs[f_b] - freqs[f_a]) <= freq_band_hz:
                        candidates.append((t_b, f_b, amp_b, dt))
                else:
                    # no frequency restriction
                    candidates.append((t_b, f_b, amp_b, dt))

            j += 1

        if not candidates:
            continue

        # sort by time (earlier first), then by amplitude (stronger first)
        candidates.sort(key=lambda x: (x[0], -x[2]))
        candidates = candidates[:fan_out]

        # build hashes: (f_anchor_bin, f_target_bin, dt_frames)
        for (t_b, f_b, amp_b, dt) in candidates:
            hashes.append((f_a, f_b, dt))

    return hashes

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
    #hop_length = 512  # default when using n_fft=2048 with librosa.stft
    # Desire 30 fps, where #fps = sample_rate / hop_length => hop_length = sample_rate / 30 
    # Thus, hop_length = 11025 / 30 = 367.5 ~ 368
    hop_length = 368

    stft = librosa.stft(signal, n_fft = 2048, hop_length=hop_length)
    spectrogram = np.abs(stft) # since stft is a complex number, we take the magnitude (real part))

    n_freq_bins, n_time_bins = spectrogram.shape
    # spectrogram has y-axis the values of the frequency bins: k in (0, 1, 2, ..., n_freq_bins-1) 
    # only to display they are converted to Hz: F(k) = k * sample_rate / n_fft
    print("freq bins:", n_freq_bins, "time bins:", n_time_bins)

    peaks = find_peaks(spectrogram, n_time_bins, n_freq_bins, bands)
    print("Total peaks found:", len(peaks))
    print(peaks[:10])
    plot_peaks = peaks[::100] # plot only every 100th peak for visibility
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)

    hashes = build_hashes(
        peaks,
        freqs,
        fan_out=5,            # 5 target points per anchor
        dt_min_frames=1,
        dt_max_frames=30,     # ≈ 1 second ahead at ~30 fps
        freq_band_hz=None     # start with no frequency restriction
    )

    print("Total hashes built:", len(hashes)) 
    print(hashes[:10])
    # 328620 as expected since #peaks = 65730 and 5 target points per anchor, but the last time frame can't have targets
    return
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

def main():
    signal, sample_rate = sf.read(Path('data') / 'windowlicker.flac')
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