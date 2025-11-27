from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm

from .config import BANDS, N_FFT, TARGET_SR
from .db import load_db
from .audio import load_audio, extract_spectrogram, find_peaks
from .hashing import build_hashes

def recognize_song(
    audio_path: Path,
    hash_db_path: str,
    songs_db_path: str,
):
    with tqdm(total=5, desc="Recognizing song", unit="step") as pbar:
        # 1. load DBs
        hash_table = load_db(hash_db_path)
        song_table = load_db(songs_db_path)
        pbar.update(1)

        # 2. process query audio
        signal, sample_rate = load_audio(audio_path)
        spectrogram = extract_spectrogram(signal, sample_rate)
        pbar.update(1)

        # 3. extract peaks
        peaks = find_peaks(spectrogram, BANDS)
        freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=N_FFT)
        pbar.update(1)

        # 4. build hashes
        fingerprints = build_hashes(peaks, freqs)
        pbar.update(1)

        # 5. match hashes
        matching_pairs = defaultdict(list)
        previous_hashes = set()

        for h, _, t_anchor in fingerprints:
            h = np.uint32(h)
            if h in previous_hashes:
                continue
            previous_hashes.add(h)
            if h in hash_table:
                matches = hash_table[h]
                previous_songs = set()
                for (song_id, t_anchor_match) in matches:
                    matching_pairs[song_id].append((t_anchor, t_anchor_match))
                    if song_id in previous_songs: continue
                    previous_songs.add(song_id)

        best_song_id = None
        best_score = 0

        for song_id, pairs in matching_pairs.items():
            offsets = [t_db - t_q for (t_q, t_db) in pairs]
            counts = Counter(offsets)
            """window = 10  # number of bins in the aggregation window
            kernel = np.ones(window, dtype=float)
            aggregated = np.convolve(list(counts.values()), kernel, mode='same')  # S[k]
            best_bin = np.argmax(aggregated)
            score = aggregated[best_bin]"""
            _, score = counts.most_common(1)[0]
            if score > best_score:
                best_score = score
                best_song_id = song_id
        pbar.update(1)

    invert_song_table = {v: k for k, v in song_table.items()}
    best_song_name = invert_song_table.get(best_song_id, None)
    return best_song_name, best_score, matching_pairs.get(best_song_id, [])
