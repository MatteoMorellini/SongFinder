import numpy as np
from typing import List, Tuple
from .config import FUZ_FACTOR

Peak = Tuple[int, int, float]
Fingerprint = Tuple[int, int, int]

def _quantize(x: int, fuzz: int = FUZ_FACTOR) -> int:
    """Round down to nearest multiple of fuzz."""
    return x - (x % fuzz)

def _hash_triplet(f_anchor: int, f_target: int, dt: int,
                  fuzz: int = FUZ_FACTOR) -> int:
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

def build_hashes_old(peaks, freqs, song_id=0, fan_out=5,
                 dt_min_frames=1, dt_max_frames=30,
                 freq_band_hz=(40, 0.4)) -> List[Fingerprint]:
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
        if freq_band_hz is not None:
            delta_f = freq_band_hz[0] + freq_band_hz[1] * freqs[f_a]
        # collect candidates in the target zone ahead of this anchor
        candidates = []
        j = i + 1
        while j < n_peaks:
            t_b, f_b, amp_b = peaks[j]
            dt = t_b - t_a
            if dt > dt_max_frames:
                break
            if dt >= dt_min_frames:
                if freq_band_hz is not None:
                    # check frequency band constraint
                    if abs(freqs[f_b] - freqs[f_a]) <= delta_f:
                        candidates.append((t_b, f_b, amp_b, dt))
                else:
                    # no frequency band constraint
                    candidates.append((t_b, f_b, amp_b, dt))
            j += 1

        if not candidates:
            continue

        # sort candidates by amplitude descending
        candidates.sort(key=lambda x: -x[2])
        # pick top fan_out strongest candidates
        strongest = candidates[:fan_out]
    
        if song_id is not None:
            for (_, f_b, _, dt) in strongest:
                h = _hash_triplet(f_a, f_b, dt)
                fingerprints.append((np.uint32(h), song_id, t_a))

    return fingerprints

def add_hashes_to_table(table, fingerprints):
    """
    table: existing dict[uint32 -> list[(song_id, t_anchor)]]
    fingerprints: iterable of (hash32, song_id, t_anchor)
                  where hash32 is np.uint32
    returns: dict[uint32 -> list[(song_id, t_anchor)]]
    """

    for h, song_id, t_anchor in fingerprints:
        h = np.uint32(h)
        
        if h not in table:
            table[h] = [(song_id, t_anchor)]
        else:
            table[h].append((song_id, t_anchor))


def build_hashes(peaks, freqs, song_id=0, fan_out=5,
                 dt_min_frames=1, dt_max_frames=30,
                 freq_band_hz=(40, 0.4)):
    """
    Optimized version:
    - avoids building/sorting a full candidates list per anchor (keeps only top fan_out online)
    - performs the frequency band constraint in bin-domain (exact for linear FFT bins)
    - inlines hashing + fast quantization for FUZ_FACTOR=2
    """
    fingerprints = []
    append = fingerprints.append
    n_peaks = len(peaks)
    if n_peaks == 0:
        return fingerprints

    # FFT bin spacing in Hz (freqs are linear for STFT rfft bins)
    # librosa.fft_frequencies and np.fft.rfftfreq give the same linear spacing
    bin_hz = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    # Precompute constants for freq band constraint:
    # Original: delta_f = base + slope * freqs[f_a]
    # Using freqs[f_a] = f_a * bin_hz, and converting to bins:
    # abs(f_b - f_a) <= delta_f / bin_hz = (base/bin_hz) + slope * f_a
    if freq_band_hz is not None:
        base, slope = freq_band_hz
        base_over = base / bin_hz

    # Fast quantize (your FUZ_FACTOR is 2)
    fuzz = FUZ_FACTOR
    if fuzz == 2:
        def q(x: int) -> int:
            return x & ~1  # round down to nearest multiple of 2
    else:
        def q(x: int) -> int:
            return x - (x % fuzz)

    abs_ = abs  # local binding is a tiny speed win

    for i in range(n_peaks):
        t_a, f_a, _amp_a = peaks[i]

        # dynamic band in bins (exact for linear FFT bins)
        if freq_band_hz is not None:
            delta_bins = base_over + slope * f_a

        # Keep only top `fan_out` by amplitude without sorting all candidates
        best_n = 0
        best_amp = [0.0] * fan_out
        best_f = [0] * fan_out
        best_dt = [0] * fan_out
        min_pos = 0
        min_amp = float("inf")

        j = i + 1
        while j < n_peaks:
            t_b, f_b, amp_b = peaks[j]
            dt = t_b - t_a
            if dt > dt_max_frames:
                break

            if dt >= dt_min_frames:
                if freq_band_hz is None or abs_(f_b - f_a) <= delta_bins:
                    if best_n < fan_out:
                        best_amp[best_n] = amp_b
                        best_f[best_n] = f_b
                        best_dt[best_n] = dt
                        if amp_b < min_amp:
                            min_amp = amp_b
                            min_pos = best_n
                        best_n += 1
                    else:
                        if amp_b > min_amp:
                            best_amp[min_pos] = amp_b
                            best_f[min_pos] = f_b
                            best_dt[min_pos] = dt
                            # recompute current minimum among the kept top-k (fan_out is small)
                            min_amp = best_amp[0]
                            min_pos = 0
                            for k in range(1, fan_out):
                                if best_amp[k] < min_amp:
                                    min_amp = best_amp[k]
                                    min_pos = k
            j += 1

        if best_n == 0:
            continue

        # Sort only the selected top fan_out candidates (<=5 items): cheap
        order = list(range(best_n))
        order.sort(key=lambda k: -best_amp[k])

        if song_id is not None:
            fa_shift = (q(f_a) << 22)
            for k in order:
                fb = q(best_f[k])
                dtq = q(best_dt[k])
                h = fa_shift | (fb << 12) | dtq
                append((np.uint32(h), song_id, t_a))

    return fingerprints


