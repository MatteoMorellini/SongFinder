"""Audio file metadata extraction using mutagen."""

from pathlib import Path
from typing import Dict, Optional


def extract_metadata(audio_path: Path) -> Dict[str, Optional[str]]:
    """Extract title, artist, album from audio file tags (ID3/FLAC/OGG/M4A)."""
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
            if "TIT2" in meta.tags:
                metadata['title'] = str(meta.tags["TIT2"].text[0])
            elif "title" in meta.tags:
                metadata['title'] = str(meta.tags["title"][0]) if isinstance(meta.tags["title"], list) else str(meta.tags["title"])
            elif "\xa9nam" in meta.tags:
                metadata['title'] = str(meta.tags["\xa9nam"][0])

            if "TPE1" in meta.tags:
                metadata['artist'] = str(meta.tags["TPE1"].text[0])
            elif "artist" in meta.tags:
                metadata['artist'] = str(meta.tags["artist"][0]) if isinstance(meta.tags["artist"], list) else str(meta.tags["artist"])
            elif "\xa9ART" in meta.tags:
                metadata['artist'] = str(meta.tags["\xa9ART"][0])

            if "TALB" in meta.tags:
                metadata['album'] = str(meta.tags["TALB"].text[0])
            elif "album" in meta.tags:
                metadata['album'] = str(meta.tags["album"][0]) if isinstance(meta.tags["album"], list) else str(meta.tags["album"])
            elif "\xa9alb" in meta.tags:
                metadata['album'] = str(meta.tags["\xa9alb"][0])
    except Exception:
        pass

    return metadata
