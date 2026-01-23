"""
Base interface for song recognition approaches.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Any, Dict


class BaseSongRecognizer(ABC):
    """
    Abstract base class for song recognition systems.
    
    Both the Shazam-style and GraFP approaches implement this interface,
    allowing them to be used interchangeably for comparison and benchmarking.
    """
    
    @abstractmethod
    def index_song(self, audio_path: Path) -> None:
        """
        Add a single song to the database/index.
        
        Args:
            audio_path: Path to the audio file to index
        """
        pass
    
    @abstractmethod
    def index_folder(self, folder: Path, pattern: str = "*.flac") -> int:
        """
        Index all songs in a folder matching the given pattern.
        
        Args:
            folder: Path to folder containing audio files
            pattern: Glob pattern for audio files (default: "*.flac")
            
        Returns:
            Number of songs successfully indexed
        """
        pass
    
    @abstractmethod
    def recognize(
        self, 
        query_path: Path, 
        clip_length_sec: Optional[float] = None,
        snr_db: Optional[float] = None
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Recognize a song from an audio query.
        
        Args:
            query_path: Path to the query audio file
            clip_length_sec: Optional clip length to use (for testing with shorter clips)
            snr_db: Optional SNR in dB for noise injection (for testing robustness)
            
        Returns:
            Tuple of (song_name, confidence_score, metadata_dict)
            - song_name: Name of the recognized song, or None if not found
            - confidence_score: Confidence/match score
            - metadata_dict: Additional information (e.g., matching pairs, timing)
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save the model/database to disk.
        
        Args:
            path: Directory or file path to save to
        """
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load the model/database from disk.
        
        Args:
            path: Directory or file path to load from
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this recognition approach."""
        pass
    
    @property
    @abstractmethod
    def num_indexed_songs(self) -> int:
        """Return the number of songs currently indexed."""
        pass
