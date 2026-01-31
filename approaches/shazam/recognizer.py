from collections import defaultdict, Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import librosa
from tqdm import tqdm
import heapq
import time

from approaches.base import BaseSongRecognizer
from .config import BANDS, N_FFT, TARGET_SR, HOP_LENGTH
from .db import load_db, save_db, get_song_id
from .audio import load_audio, extract_spectrogram, find_peaks, cut_audio, inject_noise, extract_spectrogram_fast
from .hashing import build_hashes, add_hashes_to_table


class Timer:
    """Context manager for timing code blocks with optional debug printing."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.timings: Dict[str, float] = {}
        self._current_label: Optional[str] = None
        self._start_time: float = 0
    
    @contextmanager
    def measure(self, label: str):
        """Time a block of code and optionally print the result."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.timings[label] = elapsed
        if self.debug:
            print(f"{label}: {elapsed:.4f}s")
    
    def log(self, message: str):
        """Print a message only if debug mode is enabled."""
        if self.debug:
            print(message)
    
    @property
    def total(self) -> float:
        return sum(self.timings.values())


class ShazamRecognizer(BaseSongRecognizer):
    """
    Shazam-style audio fingerprinting and recognition.
    
    Uses spectral peak constellation hashing following the classic Shazam algorithm.
    """
    
    def __init__(self, db_path: str = "fingerprints.db", songs_db_path: str = "songs.db",
                 max_query_hashes: int = 500):
        """
        Initialize the Shazam recognizer.
        
        Args:
            db_path: Path to the fingerprint hash table database
            songs_db_path: Path to the song name mapping database
            max_query_hashes: Maximum number of query hashes to use (for speed)
        """
        self.db_path = db_path
        self.songs_db_path = songs_db_path
        self.hash_table: Dict[int, list] = {}
        self.song_table: Dict[str, int] = {}
        self.max_query_hashes = max_query_hashes
        self.freqs = np.fft.rfftfreq(N_FFT, d=1.0 / TARGET_SR)

        
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
        
        song_id = get_song_id(self.song_table, song_name)
        fingerprints = build_hashes(peaks, self.freqs, song_id=song_id)
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
    
    def _sample_query_hashes(self, fingerprints, max_hashes, n_bins=20):
        n = len(fingerprints)
        if n <= max_hashes:
            return fingerprints

        # Precompute times efficiently
        times = np.fromiter((fp[2] for fp in fingerprints), dtype=np.int32, count=n)
        tmin = int(times.min())
        tmax = int(times.max()) + 1  # avoid div by 0

        # Assign each fp to a time bin in ONE pass
        denom = max(1, (tmax - tmin))
        bin_idx = ((times - tmin) * n_bins) // denom
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        bins = [[] for _ in range(n_bins)]
        for fp, b in zip(fingerprints, bin_idx):
            bins[int(b)].append(fp)

        per_bin = max(1, max_hashes // n_bins)

        # Key: rarity in DB (short posting list = better)
        def rarity(fp):
            return len(self.hash_table.get(np.uint32(fp[0]), ()))

        chosen = []
        for bin_fps in bins:
            if not bin_fps:
                continue
            # nsmallest is faster than sorting whole bin when you only need k
            chosen.extend(heapq.nsmallest(per_bin, bin_fps, key=rarity))

        # Fill remaining slots with rarest overall (avoid O(n*k) checks)
        if len(chosen) < max_hashes:
            chosen_set = set(chosen)  # tuples are hashable
            remaining = [fp for fp in fingerprints if fp not in chosen_set]
            chosen.extend(heapq.nsmallest(max_hashes - len(chosen), remaining, key=rarity))

        return chosen[:max_hashes]


    
    def recognize(
        self, 
        query_path: Path, 
        clip_length_sec: Optional[float] = None,
        snr_db: Optional[float] = None,
        top_songs_entropy: Optional[int] = 10,
        debug: bool = False,
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:

        print(debug)
    
        """
        Recognize a song from an audio query.
        
        Args:
            query_path: Path to the audio file to recognize
            clip_length_sec: Optional clip length in seconds
            snr_db: Optional SNR for noise injection
            top_songs_entropy: Number of top songs for entropy calculation
            debug: If True, print timing information for each step
        
        Returns:
            Tuple of (song_name, score, metadata)
        """
        timer = Timer(debug=debug)
        
        # Load and preprocess query audio
        with timer.measure("Load audio"):
            signal, sample_rate = load_audio(query_path)
        
        if clip_length_sec is not None:
            with timer.measure("Cut audio"):
                signal = cut_audio(signal, sample_rate, clip_length_sec)
        
        if snr_db is not None:
            with timer.measure("Inject noise"):
                signal = inject_noise(signal, snr_db)
        
        # Extract fingerprints from query
        with timer.measure("Extract spectrogram"):
            spectrogram = extract_spectrogram_fast(signal, sample_rate)
        
        with timer.measure("Find peaks"):
            peaks = find_peaks(spectrogram, BANDS)
        
        with timer.measure("Build hashes"):
            fingerprints = build_hashes(peaks, self.freqs)
        
        # Sample query hashes for faster lookup
        with timer.measure("Sample hashes"):
            sampled_fingerprints = self._sample_query_hashes(fingerprints, self.max_query_hashes)
        timer.log(f"  Hashes: {len(fingerprints)} -> {len(sampled_fingerprints)}")
        
        # Match against database and vote in single pass
        with timer.measure("Hash matching and voting"):
            # song_id -> {offset -> count}
            offset_votes = defaultdict(lambda: defaultdict(int))
            num_matches = 0
            num_db_hits = 0
            
            for h, _, t_anchor in sampled_fingerprints:
                h = np.uint32(h)
                
                if h in self.hash_table:
                    num_db_hits += 1
                    for (song_id, t_anchor_match) in self.hash_table[h]:
                        offset = t_anchor_match - t_anchor
                        offset_votes[song_id][offset] += 1
                        num_matches += 1

        timer.log(f"  Query hashes checked: {len(sampled_fingerprints)}")
        timer.log(f"  Database hits: {num_db_hits}")
        timer.log(f"  Total matches: {num_matches}")
        timer.log(f"  Candidate songs: {len(offset_votes)}")
        
        # Find best match by getting peak offset count for each song
        with timer.measure("Scoring preparation"):
            song_scores = {}
            for song_id, offset_counts in offset_votes.items():
                # Get the maximum vote count (most common offset)
                song_scores[song_id] = max(offset_counts.values())
        
        with timer.measure("Softmax scoring"):
            if song_scores:
                songs = list(song_scores.keys())
                scores = np.array(list(song_scores.values()), dtype=np.float64)

                best_idx = int(np.argmax(scores))
                best_song_id = songs[best_idx]

                # Take top-K scores (highest first)
                top_idx = np.argsort(scores)[::-1][:top_songs_entropy]
                top_scores = scores[top_idx]

                sum_s = float(np.sum(top_scores))
                if sum_s <= 0 or len(top_scores) == 0:
                    best_confidence = 0.0
                else:
                    p = top_scores / sum_s
                    eps = 1e-12
                    H = -float(np.sum(p * np.log(p + eps)))
                    H_norm = H / np.log(len(p)) if len(p) > 1 else 0.0
                    best_confidence = float(1.0 - H_norm)
                    best_confidence = max(0.0, min(1.0, best_confidence))

            else:
                best_song_id = None
                best_confidence = 0.0
        
        # Get song name from ID
        with timer.measure("Lookup"):
            invert_song_table = {v: k for k, v in self.song_table.items()}
            best_song_name = invert_song_table.get(best_song_id)
        
        timer.log(f"Total recognition time: {timer.total:.4f}s")
        
        metadata = {
            "num_query_hashes": len(fingerprints),
            "num_sampled_hashes": len(sampled_fingerprints),
            "num_matched_hashes": num_db_hits,
            "num_total_matches": num_matches,
            "num_candidate_songs": len(offset_votes),
            "best_song_offset_distribution": dict(offset_votes.get(best_song_id, {})) if best_song_id else {},
            "best_song_score": song_scores.get(best_song_id, 0),
            "timings": timer.timings,
            "total_time": timer.total,
        }
        
        return best_song_name, float(best_confidence), metadata