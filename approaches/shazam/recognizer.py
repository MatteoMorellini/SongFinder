"""
Shazam-style song recognizer implementing the BaseSongRecognizer interface.
"""

from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import librosa
from tqdm import tqdm

from approaches.base import BaseSongRecognizer
from .config import BANDS, N_FFT, TARGET_SR, HOP_LENGTH
from .db import load_db, save_db, get_song_id
from .audio import load_audio, extract_spectrogram, find_peaks, cut_audio, inject_noise
from .hashing import build_hashes, add_hashes_to_table


class ShazamRecognizer(BaseSongRecognizer):
    """
    Shazam-style audio fingerprinting and recognition.
    
    Uses spectral peak constellation hashing following the classic Shazam algorithm.
    """
    
    def __init__(self, db_path: str = "fingerprints.db", songs_db_path: str = "songs.db"):
        """
        Initialize the Shazam recognizer.
        
        Args:
            db_path: Path to the fingerprint hash table database
            songs_db_path: Path to the song name mapping database
        """
        self.db_path = db_path
        self.songs_db_path = songs_db_path
        self.hash_table: Dict[int, list] = {}
        self.song_table: Dict[str, int] = {}
        
    @property
    def name(self) -> str:
        return "Shazam"
    
    @property
    def num_indexed_songs(self) -> int:
        return len(self.song_table)
    
    def load(self, path: Optional[Path] = None) -> None:
        """Load databases from disk."""
        db_path = str(path / "fingerprints.db") if path else self.db_path
        songs_path = str(path / "songs.db") if path else self.songs_db_path
        self.hash_table = load_db(db_path)
        self.song_table = load_db(songs_path)
        
    def save(self, path: Optional[Path] = None) -> None:
        """Save databases to disk."""
        db_path = str(path / "fingerprints.db") if path else self.db_path
        songs_path = str(path / "songs.db") if path else self.songs_db_path
        save_db(db_path, self.hash_table)
        save_db(songs_path, self.song_table)
    
    def index_song(self, audio_path: Path) -> None:
        """Add a single song to the database."""
        song_name = audio_path.stem
        if song_name in self.song_table:
            return  # Already indexed
            
        signal, sr = load_audio(audio_path)
        spectrogram = extract_spectrogram(signal, sr)
        peaks = find_peaks(spectrogram, BANDS)
        freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=N_FFT)
        
        song_id = get_song_id(self.song_table, song_name)
        fingerprints = build_hashes(peaks, freqs, song_id=song_id)
        add_hashes_to_table(self.hash_table, fingerprints)
    
    def index_folder(self, folder: Path, pattern: str = "*.flac") -> int:
        """Index all songs in a folder."""
        audio_paths = list(folder.glob(pattern))
        count = 0
        for audio_path in tqdm(audio_paths, desc="Indexing songs", unit='song'):
            initial_count = len(self.song_table)
            self.index_song(audio_path)
            if len(self.song_table) > initial_count:
                count += 1
        return count
    
    def recognize(
        self, 
        query_path: Path, 
        clip_length_sec: Optional[float] = None,
        snr_db: Optional[float] = None
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
    
        """
        Recognize a song from an audio query.
        
        Returns:
            Tuple of (song_name, score, metadata)
        """
        # Load and preprocess query audio
        signal, sample_rate = load_audio(query_path)
        
        if clip_length_sec is not None:
            signal = cut_audio(signal, sample_rate, clip_length_sec)
        if snr_db is not None:
            signal = inject_noise(signal, snr_db)
        
        # Extract fingerprints from query
        spectrogram = extract_spectrogram(signal, sample_rate)
        peaks = find_peaks(spectrogram, BANDS)
        freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=N_FFT)
        fingerprints = build_hashes(peaks, freqs)
        
        # Match against database
        matching_pairs = defaultdict(list)
        seen_hashes = set()
        
        for h, _, t_anchor in fingerprints:
            h = np.uint32(h)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            
            if h in self.hash_table:
                for (song_id, t_anchor_match) in self.hash_table[h]:
                    matching_pairs[song_id].append((t_anchor, t_anchor_match))
        
        # Find best match using time offset voting
        song_scores = {}
        for song_id, pairs in matching_pairs.items():
            offsets = [t_db - t_q for (t_q, t_db) in pairs]
            counts = Counter(offsets)
            if counts:
                _, score = counts.most_common(1)[0]  # Get peak score
                song_scores[song_id] = score
            
        if song_scores:
            # Extract songs and scores in order
            songs = list(song_scores.keys())
            scores = np.array(list(song_scores.values()), dtype=np.float64)
            
            # Apply softmax
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / np.sum(exp_scores)
            
            # Get song with highest probability (= highest score)
            best_idx = np.argmax(probabilities)
            best_song_id = songs[best_idx]
            best_confidence = probabilities[best_idx]
        else:
            best_song_id = None
            best_confidence = 0.0
        
        # Get song name from ID
        invert_song_table = {v: k for k, v in self.song_table.items()}
        best_song_name = invert_song_table.get(best_song_id)
        
        metadata = {
            "matching_pairs": matching_pairs.get(best_song_id, []),
            "num_query_hashes": len(fingerprints),
            "num_matched_hashes": len(seen_hashes & set(self.hash_table.keys())),
        }
        
        return best_song_name, float(best_confidence), metadata
