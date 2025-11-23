import soundfile as sf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
from typing import List, Tuple, Dict

# type aliases
Peak = Tuple[int, int, float]         # (t_frame, f_bin, amplitude)
Fingerprint = Tuple[int, int, int]    # (hash32, song_id, t_anchor_frame)

FUZ_FACTOR = 2  # absorb small variations in frequency / time: 43 → 42, 21 → 20, etc.

def _quantize(x: int, fuzz: int = FUZ_FACTOR) -> int:
    """Round down to nearest multiple of fuzz."""
    return x - (x % fuzz)


def _hash_triplet(f_anchor: int, f_target: int, dt: int,
                  fuzz: int = 2) -> int:
    """
    Pack (f_anchor, f_target, dt) into a 32-bit integer.

    Layout (MSB → LSB):
        [10 bits f_anchor][10 bits f_target][12 bits dt]

    Assumes:
        f_anchor, f_target < 1024
        dt < 4096
    """
    # fuzzy quantization (error-correction)
    fa = _quantize(f_anchor, fuzz)
    fb = _quantize(f_target, fuzz)
    dt = _quantize(dt, fuzz)

    # clamp to bit ranges (safety)
    fa = max(0, min(fa, 1023))
    fb = max(0, min(fb, 1023))
    dt = max(0, min(dt, 4095))

    # bit pack: fa[31:22], fb[21:12], dt[11:0]
    return (fa << 22) | (fb << 12) | dt


def build_hashes(
    peaks,
    freqs,
    fan_out=5,
    dt_min_frames=1,      # ≥ 1 frame ahead
    dt_max_frames=30*6,     # ≤ ~1s ahead if ~30 fps
    song_id=0,
    freq_band_hz=(40,0.4),    # e.g. 1200 → ±1200 Hz around anchor; None = no limit
    fuzz=FUZ_FACTOR,
):
    """
    peaks:      list of (t_frame, f_bin, amplitude), assumed sorted by t_frame
    freqs:      1D array mapping frequency-bin index → Hz (librosa.fft_frequencies)
    sample_rate, hop_length: used only for reference / debugging
    fan_out:    max number of target points per anchor # ! interesting hyperparameter to twist in experiments
    returns:    list of hashes and metadata
                hashes: list of (f_anchor_bin, f_target_bin, dt_frames)
    """
    fingerprints: List[Fingerprint] = []
    n_peaks = len(peaks)

    for i in range(n_peaks):
        t_a, f_a, amp_a = peaks[i]
        delta_f = freq_band_hz[0] + freq_band_hz[1] * freqs[f_a]
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
                    if abs(freqs[f_b] - freqs[f_a]) <= delta_f:
                        candidates.append((t_b, f_b, amp_b, dt))
                else:
                    # no frequency restriction
                    candidates.append((t_b, f_b, amp_b, dt))

            j += 1

        if not candidates:
            continue

        # Sort candidates by amplitude descending
        candidates.sort(key=lambda x: -x[2])

        # Pick top fan_out strongest peaks
        strongest = candidates[:fan_out]

        
        for (_, f_b, _, dt) in strongest:
            h = _hash_triplet(f_a, f_b, dt, fuzz=fuzz)
            #print(f"Anchor t={t_a}, f={f_a} ({freqs[f_a]:.1f} Hz) -> Target f={f_b} ({freqs[f_b]:.1f} Hz), dt={dt} frames => Hash={h:08x}")
            fingerprints.append((np.uint32(h), song_id, t_a))
    return fingerprints

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

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)

    fingerprints = build_hashes(
        peaks,
        freqs,
        fan_out=5,            # 5 target points per anchor
        dt_min_frames=1,
        dt_max_frames=30,     # ≈ 1 second ahead at ~30 fps
    )

    print("Total hashes built:", len(fingerprints)) 
    # 328620 as expected since #peaks = 65730 and 5 target points per anchor, but the last time frame can't have targets # ? check

    plot_peaks = peaks[::200] # plot only every 100th peak for visibility
    
    
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
    signal, sample_rate = sf.read(Path('data') / 'isthisit.flac')
    print(f"Sample Rate: {sample_rate}")

    # Define frequency bands (in terms of frequency bin indices)
    # n_fft = 2048 -> freq bins = 1025 (0 to 1024) but we will limit to ~5kHz
    bands = [
        (0, 10),      # very low
        (11, 20),     # low
        (21, 40),     # low-mid
        (41, 80),     # mid
        (81, 160),    # mid-high
        (161, 511)    # high
    ]

    plot_spectrogram_and_save(signal, sample_rate, Path('imgs') / 'spectrogram.png', bands)




if __name__ == '__main__':
    main()