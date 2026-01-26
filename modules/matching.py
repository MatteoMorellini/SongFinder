from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf
from .config import BANDS, N_FFT, TARGET_SR, DB_PATH, SONGS_DB_PATH, HOP_LENGTH
from .db import load_db
from .audio import load_audio, extract_spectrogram, find_peaks, cut_audio, inject_noise, plot_spectrogram_and_save
from .hashing import build_hashes
from .graph import build_graph
import os
import torch
from torch_geometric.data import Data, Dataset
# Pre-load databases
hash_table = load_db(DB_PATH)
song_table = load_db(SONGS_DB_PATH)

def recognize_song(audio_path: Path, clip_length_sec = None, snr_db = None):
    with tqdm(total=4, desc="Recognizing song", unit="step") as pbar:

        # 1. process query audio
        signal, sample_rate = load_audio(audio_path)
        if clip_length_sec is not None:
            signal = cut_audio(signal, sample_rate, clip_length_sec)
        if snr_db is not None:
            signal = inject_noise(signal, snr_db)
            
        sf.write(f"temp_{snr_db}dB_{clip_length_sec}sec.wav", signal, sample_rate)

        spectrogram = extract_spectrogram(signal, sample_rate)
        os.makedirs('imgs', exist_ok=True)
        pbar.update(1)

        # 2. extract peaks
        peaks = find_peaks(spectrogram, BANDS)
        freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=N_FFT)
        #plot_spectrogram_and_save(spectrogram, TARGET_SR, HOP_LENGTH, peaks, freqs, Path('imgs') / f'{audio_path.stem}_{snr_db}dB_{clip_length_sec}sec.png')

        pbar.update(1)

        # 3. build hashes
        fingerprints, edges = build_hashes(peaks, freqs)
        pbar.update(1)

        print(edges[-1])

        dataset = build_graph(peaks, edges)
        torch.manual_seed(12345)
        dataset = dataset.shuffle()

        train_dataset = dataset[:150]
        test_dataset = dataset[150:]

        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of test graphs: {len(test_dataset)}')

        # 4. match hashes
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
