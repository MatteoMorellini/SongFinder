from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import librosa
from tqdm import tqdm
import heapq

from approaches.base import BaseSongRecognizer
from .config import BANDS, N_FFT, TARGET_SR, HOP_LENGTH
from .db import load_db, save_db, get_song_id
from .audio import load_audio, extract_spectrogram, find_peaks, cut_audio, inject_noise, extract_spectrogram_fast
from .hashing import build_hashes, add_hashes_to_table


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
        snr_db: Optional[float] = None
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
    
        """
        Recognize a song from an audio query.
        
        Returns:
            Tuple of (song_name, score, metadata)
        """
        import time
        
        timings = {}
        
        # Load and preprocess query audio
        t0 = time.perf_counter()
        signal, sample_rate = load_audio(query_path)
        timings['load_audio'] = time.perf_counter() - t0
        print(f"Load audio: {timings['load_audio']:.4f}s")
        
        if clip_length_sec is not None:
            t0 = time.perf_counter()
            signal = cut_audio(signal, sample_rate, clip_length_sec)
            timings['cut_audio'] = time.perf_counter() - t0
            print(f"Cut audio: {timings['cut_audio']:.4f}s")
        
        if snr_db is not None:
            t0 = time.perf_counter()
            signal = inject_noise(signal, snr_db)
            timings['inject_noise'] = time.perf_counter() - t0
            print(f"Inject noise: {timings['inject_noise']:.4f}s")
        
        # Extract fingerprints from query
        t0 = time.perf_counter()
        spectrogram = extract_spectrogram_fast(signal, sample_rate)
        timings['extract_spectrogram'] = time.perf_counter() - t0
        print(f"Extract spectrogram: {timings['extract_spectrogram']:.4f}s")
        
        t0 = time.perf_counter()
        peaks = find_peaks(spectrogram, BANDS)
        timings['find_peaks'] = time.perf_counter() - t0
        print(f"Find peaks: {timings['find_peaks']:.4f}s")
        
        t0 = time.perf_counter()
        fingerprints = build_hashes(peaks, self.freqs)
        timings['build_hashes'] = time.perf_counter() - t0
        print(f"Build hashes: {timings['build_hashes']:.4f}s")
        
        # Sample query hashes for faster lookup
        t0 = time.perf_counter()
        sampled_fingerprints = self._sample_query_hashes(fingerprints, self.max_query_hashes)
        timings['sample_hashes'] = time.perf_counter() - t0
        print(f"Sample hashes: {len(fingerprints)} -> {len(sampled_fingerprints)} ({timings['sample_hashes']:.4f}s)")
        
        # Match against database and vote in single pass
        t0 = time.perf_counter()
        # song_id -> {offset -> count}
        offset_votes = defaultdict(lambda: defaultdict(int))
        seen_hashes = set()
        num_matches = 0
        num_db_hits = 0
        
        for h, _, t_anchor in sampled_fingerprints:
            h = np.uint32(h)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            
            if h in self.hash_table:
                num_db_hits += 1
                for (song_id, t_anchor_match) in self.hash_table[h]:
                    offset = t_anchor_match - t_anchor
                    offset_votes[song_id][offset] += 1
                    num_matches += 1
        
        timings['hash_matching_and_voting'] = time.perf_counter() - t0
        print(f"Hash matching and voting: {timings['hash_matching_and_voting']:.4f}s")
        print(f"  Query hashes checked: {len(sampled_fingerprints)}")
        print(f"  Database hits: {num_db_hits}")
        print(f"  Total matches: {num_matches}")
        print(f"  Candidate songs: {len(offset_votes)}")
        
        # Find best match by getting peak offset count for each song
        t0 = time.perf_counter()
        song_scores = {}
        for song_id, offset_counts in offset_votes.items():
            # Get the maximum vote count (most common offset)
            song_scores[song_id] = max(offset_counts.values())
        
        timings['scoring_preparation'] = time.perf_counter() - t0
        print(f"Scoring preparation: {timings['scoring_preparation']:.4f}s")
        
        t0 = time.perf_counter()
        if song_scores:
            # Convert to arrays for vectorized operations
            songs = np.array(list(song_scores.keys()))
            scores = np.array(list(song_scores.values()), dtype=np.float64)
            
            # Apply softmax
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / np.sum(exp_scores)
            
            # Get song with highest probability (= highest score)
            best_idx = np.argmax(probabilities)
            best_song_id = songs[best_idx]
            best_confidence = float(probabilities[best_idx])
        else:
            best_song_id = None
            best_confidence = 0.0
        
        timings['softmax_scoring'] = time.perf_counter() - t0
        print(f"Softmax scoring: {timings['softmax_scoring']:.4f}s")
        
        # Get song name from ID
        t0 = time.perf_counter()
        invert_song_table = {v: k for k, v in self.song_table.items()}
        best_song_name = invert_song_table.get(best_song_id)
        timings['lookup'] = time.perf_counter() - t0
        print(f"Lookup: {timings['lookup']:.4f}s")
        
        total_time = sum(timings.values())
        print(f"Total recognition time: {total_time:.4f}s")
        
        metadata = {
            "num_query_hashes": len(fingerprints),
            "num_sampled_hashes": len(sampled_fingerprints),
            "num_matched_hashes": num_db_hits,
            "num_total_matches": num_matches,
            "num_candidate_songs": len(offset_votes),
            "best_song_offset_distribution": dict(offset_votes.get(best_song_id, {})) if best_song_id else {},
            "best_song_score": song_scores.get(best_song_id, 0),
            "timings": timings,
            "total_time": total_time,
        }
        
        return best_song_name, float(best_confidence), metadata