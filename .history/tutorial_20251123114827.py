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

def generate_fingerprints(
    peaks: List[Peak],
    song_id: int,
    fan_out: int = 5,
    dt_min: int = 1,
    dt_max: int = 30,
    fuzz: int = 2,
) -> List[Fingerprint]:
    """
    Generate Shazam-style fingerprints from a list of peaks.

    peaks:   list of (t_frame, f_bin, amplitude), ideally sorted by t_frame
    song_id: integer ID of the track
    fan_out: max number of target peaks per anchor
    dt_min:  minimum Δt (in frames) to consider
    dt_max:  maximum Δt (in frames) to consider
    fuzz:    quantization step for error-correction

    returns: list of (hash32, song_id, t_anchor_frame)
    """

    # ensure peaks sorted by time
    peaks = sorted(peaks, key=lambda p: p[0])

    fingerprints: List[Fingerprint] = []
    n = len(peaks)

    for i in range(n):
        t_a, f_a, amp_a = peaks[i]

        candidates = []
        # look forward in time only
        for j in range(i + 1, n):
            t_b, f_b, amp_b = peaks[j]
            dt = t_b - t_a

            if dt < dt_min:
                continue
            if dt > dt_max:
                break  # beyond target window

            candidates.append((t_b, f_b, amp_b, dt))

        if not candidates:
            continue

        candidates.sort(key=lambda x: -x[2]) # sort by amplitude descending
        candidates = candidates[:fan_out]

        for (_, f_b, _, dt) in candidates:
            h = _hash_triplet(f_a, f_b, dt, fuzz=fuzz)
            print("Hash:", h, "Anchor freq bin:", f_a, "Target freq bin:", f_b, "Delta t:", dt)
            break
            fingerprints.append((h, song_id, t_a))

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

    hashes = build_hashes(
        peaks,
        freqs,
        fan_out=5,            # 5 target points per anchor
        dt_min_frames=1,
        dt_max_frames=30,     # ≈ 1 second ahead at ~30 fps
        freq_band_hz=None     # start with no frequency restriction
    )

    print("Total hashes built:", len(hashes)) 
    print(hashes[10000:10050])

    plot_peaks = peaks[::100] # plot only every 100th peak for visibility
    
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
        (11, 20),     # low
        (21, 40),     # low-mid
        (41, 80),     # mid
        (81, 160),    # mid-high
        (161, 511)    # high
    ]

    plot_spectrogram_and_save(signal, sample_rate, Path('imgs') / 'spectrogram.png', bands)




if __name__ == '__main__':
    main()