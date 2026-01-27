from pathlib import Path
import librosa
import os
from tqdm import tqdm
from .config import BANDS, N_FFT, PLOT_SPECTROGRAM, TARGET_SR, HOP_LENGTH
from .db import load_db, save_db, get_song_id
from .audio import load_audio, extract_spectrogram, find_peaks, plot_spectrogram_and_save
from .hashing import build_hashes, add_hashes_to_table

import warnings
warnings.filterwarnings("ignore")


def index_song(audio_path: Path, hash_table, song_table):
    song_name = audio_path.stem  # filename without extension
    if song_name in song_table:
        #print(f"Song '{song_name}' is already indexed. Skipping.")
        return
    #print(f"Processing {song_name}")
    signal, sr = load_audio(audio_path)
    if signal is None or sr is None:
        return
    spectrogram = extract_spectrogram(signal, sr)
    peaks = find_peaks(spectrogram, BANDS)
    freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=N_FFT)
        
    song_id = get_song_id(song_table, song_name)

    fingerprints = build_hashes(peaks, freqs, song_id=song_id)
    add_hashes_to_table(hash_table, fingerprints)

    if not PLOT_SPECTROGRAM:
        return
    os.makedirs('imgs', exist_ok=True)
    plot_spectrogram_and_save(spectrogram, TARGET_SR, HOP_LENGTH, peaks, freqs, Path('imgs') / f'{song_name}.png')

def index_folder(
    folder: Path,
    hash_db_path: str,
    songs_db_path: str,
    pattern: str = "*.flac",
):
    hash_table = load_db(hash_db_path)
    song_table = load_db(songs_db_path)

    audio_paths = []

    for p in Path(folder).glob(pattern):
        try:
            duration = librosa.get_duration(path=p)
            if duration >= 5:
                audio_paths.append(p)
        except Exception:
            pass 
    audio_paths = list(folder.glob(pattern))
    print(audio_paths[:10])
    for audio_path in tqdm(audio_paths, desc="Indexing songs", unit='song'):
        index_song(audio_path, hash_table, song_table)

    save_db(hash_db_path, hash_table)
    save_db(songs_db_path, song_table)