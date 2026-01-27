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

def build_hashes(peaks, freqs, song_id=0, fan_out=5,
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

