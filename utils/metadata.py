"""
Shared utilities for extracting audio file metadata.
"""
from pathlib import Path
from typing import Dict, Optional


def extract_metadata(audio_path: Path) -> Dict[str, Optional[str]]:
    """
    Extract metadata from audio file tags.

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with keys: filename, title, artist, album
        If tags not found, title/artist/album will be None
    """
    from mutagen import File as MutagenFile

    metadata = {
        'filename': audio_path.stem,
        'title': None,
        'artist': None,
        'album': None,
    }

    try:
        meta = MutagenFile(audio_path)
        if meta is not None and meta.tags is not None:
            # Extract title
            if "TIT2" in meta.tags:  # MP3/ID3
                metadata['title'] = str(meta.tags["TIT2"].text[0])
            elif "title" in meta.tags:  # FLAC, OGG, M4A
                metadata['title'] = str(meta.tags["title"][0]) if isinstance(meta.tags["title"], list) else str(meta.tags["title"])
            elif "\xa9nam" in meta.tags:  # M4A/ALAC alternative
                metadata['title'] = str(meta.tags["\xa9nam"][0])

            # Extract artist
            if "TPE1" in meta.tags:  # MP3/ID3
                metadata['artist'] = str(meta.tags["TPE1"].text[0])
            elif "artist" in meta.tags:  # FLAC, OGG, M4A
                metadata['artist'] = str(meta.tags["artist"][0]) if isinstance(meta.tags["artist"], list) else str(meta.tags["artist"])
            elif "\xa9ART" in meta.tags:  # M4A/ALAC alternative
                metadata['artist'] = str(meta.tags["\xa9ART"][0])

            # Extract album
            if "TALB" in meta.tags:  # MP3/ID3
                metadata['album'] = str(meta.tags["TALB"].text[0])
            elif "album" in meta.tags:  # FLAC, OGG, M4A
                metadata['album'] = str(meta.tags["album"][0]) if isinstance(meta.tags["album"], list) else str(meta.tags["album"])
            elif "\xa9alb" in meta.tags:  # M4A/ALAC alternative
                metadata['album'] = str(meta.tags["\xa9alb"][0])
    except Exception:
        pass

    return metadata
