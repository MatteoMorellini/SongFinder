#!/usr/bin/env python3
"""
Index songs into the database for Shazam or GraFP.

Usage:
    # Shazam
    python scripts/index_songs.py --approach shazam --folder ~/datasets/fma_small
    
    # GraFP (requires checkpoint)
    python scripts/index_songs.py --approach grafp --folder ~/datasets/fma_small \
                                  --checkpoint path/to/model.pth \
                                  --chunk-duration 60 --chunk-overlap 30
"""

import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

def append_grafp_db(output_dir: Path, new_fp: np.ndarray, new_meta: np.ndarray, new_metadata_dict: dict = None):
    """
    Append new fingerprints to GraFP database.

    Args:
        output_dir: Output directory for database files
        new_fp: New fingerprints array
        new_meta: New metadata array (filename for each segment)
        new_metadata_dict: Dictionary mapping filename -> metadata dict (title, artist, album)
    """
    import pickle

    db_path = output_dir / "db.mm"
    shape_path = output_dir / "db_shape.npy"
    meta_path = output_dir / "db_metadata.npy"
    metadata_table_path = output_dir / "metadata_table.pkl"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or initialize metadata_table
    if metadata_table_path.exists():
        with open(metadata_table_path, 'rb') as f:
            metadata_table = pickle.load(f)
    else:
        metadata_table = {}

    # Update metadata_table with new entries
    if new_metadata_dict:
        metadata_table.update(new_metadata_dict)

    if db_path.exists() and shape_path.exists() and meta_path.exists():
        old_shape = tuple(np.load(shape_path))
        old_meta = np.load(meta_path, allow_pickle=True)
        old_db = np.memmap(db_path, dtype="float32", mode="r", shape=old_shape)

        # --- normalize metadata shapes to 1D ---
        old_meta = np.asarray(old_meta, dtype=object).reshape(-1)
        new_meta = np.asarray(new_meta, dtype=object).reshape(-1)

        # (optional but recommended) sanity checks
        if old_meta.shape[0] != old_shape[0]:
            raise ValueError(f"Metadata length ({old_meta.shape[0]}) != db rows ({old_shape[0]})")
        if new_meta.shape[0] != new_fp.shape[0]:
            raise ValueError(f"New metadata length ({new_meta.shape[0]}) != new_fp rows ({new_fp.shape[0]})")

        print(f"old meta length: {len(old_meta)}")
        print(old_meta.shape, new_meta.shape)

        # build combined
        combined_shape = (old_shape[0] + new_fp.shape[0], old_shape[1])
        combined_meta = np.concatenate([old_meta, new_meta])

        print(f"new entries: {len(new_meta)}")
        print(f"updated meta length: {len(combined_meta)}")

        # CRITICAL FIX: Copy old data to memory BEFORE truncating file
        # mode='w+' truncates the file, corrupting old_db which points to same file
        old_data_copy = np.array(old_db)  # Copy to RAM
        del old_db  # Close old memmap

        # Now safe to create new memmap (file will be truncated)
        new_db = np.memmap(db_path, dtype="float32", mode="w+", shape=combined_shape)
        new_db[:old_shape[0]] = old_data_copy
        new_db[old_shape[0]:] = new_fp
        new_db.flush()
        del new_db  # Ensure flush completes

        np.save(shape_path, np.array(combined_shape))
        np.save(meta_path, combined_meta)

    else:
        # first time
        new_db = np.memmap(db_path, dtype="float32", mode="w+", shape=new_fp.shape)
        new_db[:] = new_fp
        new_db.flush()
        del new_db  # Ensure flush completes
        np.save(shape_path, np.array(new_fp.shape))
        np.save(meta_path, new_meta)

    # Save metadata_table
    with open(metadata_table_path, 'wb') as f:
        pickle.dump(metadata_table, f)


def index_shazam(folder: Path, output_dir: Path, pattern: str):
    """Index songs using Shazam approach."""
    from approaches.shazam import ShazamRecognizer
    from tqdm import tqdm
    
    print("\n=== Shazam Indexing ===")
    
    recognizer = ShazamRecognizer()
    
    # Try to load existing
    try:
        recognizer.load(output_dir)
        print(f"Loaded existing: {recognizer.num_indexed_songs} songs")
    except:
        print("Starting fresh database")
    
    audio_files = list(folder.rglob(pattern))
    if not audio_files:
        audio_files = list(folder.rglob("*.mp3"))
    
    print(f"Found {len(audio_files)} audio files")
    
    for f in tqdm(audio_files, desc="Indexing"):
        try:
            recognizer.index_song(f)
        except Exception as e:
            print(f"Error {f.name}: {e}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    recognizer.save(output_dir)
    
    print(f"✓ Saved {recognizer.num_indexed_songs} songs to {output_dir}")
    return recognizer.num_indexed_songs


def chunk_audio(waveform: np.ndarray, sr: int, chunk_duration: float, 
                overlap_duration: float):
    """
    Split audio into overlapping chunks.
    
    Args:
        waveform: Audio waveform (1D numpy array)
        sr: Sample rate
        chunk_duration: Duration of each chunk in seconds
        overlap_duration: Overlap between consecutive chunks in seconds
    
    Returns:
        List of tuples (chunk_waveform, start_time_seconds)
    """
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap_duration * sr)
    hop_samples = chunk_samples - overlap_samples
    
    chunks = []
    start_sample = 0
    
    while start_sample < len(waveform):
        end_sample = min(start_sample + chunk_samples, len(waveform))
        chunk = waveform[start_sample:end_sample]
        
        # Only keep chunks that are at least half the target duration
        # to avoid very short chunks at the end
        min_duration = chunk_duration * 0.5
        if len(chunk) >= int(min_duration * sr):
            start_time = start_sample / sr
            chunks.append((chunk, start_time))
        
        # If we've reached the end, break
        if end_sample >= len(waveform):
            break
            
        start_sample += hop_samples
    
    return chunks


def index_grafp(folder: Path, output_dir: Path, checkpoint: str,
                config: str, device: str, pattern: str,
                chunk_duration: float = 60.0, chunk_overlap: float = 15.0):
    """
    Index songs using GraFP approach with chunking.

    Args:
        chunk_duration: Duration of each chunk in seconds (default: 60)
        chunk_overlap: Overlap between chunks in seconds (default: 30)
    """
    import torch
    import torchaudio
    import numpy as np
    import pickle
    from tqdm import tqdm
    from approaches.grafp import load_config, load_model
    from approaches.grafp.modules.transformations import AudioTransform
    from utils.augmentation import load_audio
    from utils.metadata import extract_metadata

    print("\n=== GraFP Indexing ===")
    print(f"Chunk settings: {chunk_duration}s duration, {chunk_overlap}s overlap")
    print(f"Effective hop: {chunk_duration - chunk_overlap}s")

    cfg = load_config(config)
    model = load_model(cfg, checkpoint)
    transform = AudioTransform(cfg).to(device)

    # Load existing metadata_table to check for duplicates
    metadata_table_path = output_dir / "metadata_table.pkl"
    if metadata_table_path.exists():
        with open(metadata_table_path, 'rb') as f:
            existing_metadata = pickle.load(f)
        print(f"Loaded existing metadata: {len(existing_metadata)} songs indexed")
    else:
        existing_metadata = {}

    audio_files = list(folder.rglob(pattern))
    if not audio_files:
        audio_files = list(folder.rglob("*.mp3"))

    print(f"Found {len(audio_files)} audio files")

    fingerprints = []
    metadata = []
    new_metadata_dict = {}  # filename -> metadata dict

    model.eval()
    total_chunks = 0
    skipped = 0

    for f in tqdm(audio_files, desc="Processing songs"):
        filename = f.stem

        # Check if already indexed (deduplication)
        if filename in existing_metadata:
            skipped += 1
            continue
        # Extract metadata from tags (once per song)
        song_metadata = extract_metadata(f)
        new_metadata_dict[filename] = song_metadata

        try:
            # Load and resample audio file (handles ISO Media/ALAC, avoids soxr)
            waveform_np, sr = load_audio(f, cfg['fs'])

            # Split into chunks
            chunks = chunk_audio(waveform_np, sr, chunk_duration, chunk_overlap)

            if not chunks:
                print(f"Warning: No valid chunks for {f.name}")
                continue

            # Process each chunk
            for chunk_idx, (chunk_waveform, start_time) in enumerate(chunks):
                try:
                    waveform = torch.from_numpy(chunk_waveform).float()
                    segments = transform(waveform.unsqueeze(0).to(device))

                    with torch.no_grad():
                        _, _, z, _ = model(segments, segments)

                    fingerprints.append(z.cpu().numpy())

                    # Store filename for all segments (for FAISS matching)
                    for _ in range(z.shape[0]):
                        metadata.append(filename)

                    total_chunks += 1

                except Exception as e:
                    print(f"Error processing chunk {chunk_idx} of {f.name}: {e}")
                    continue

        except Exception as e:
            print(f"Error loading {f.name}: {e}")
            continue
    
    if skipped > 0:
        print(f"Skipped {skipped} already indexed songs")

    if fingerprints:
        fp_array = np.concatenate(fingerprints).astype('float32')
        new_meta = np.array(metadata)
        append_grafp_db(output_dir, fp_array, new_meta, new_metadata_dict)

        new_songs_count = len(new_metadata_dict)
        print(f"✓ Saved {new_songs_count} new songs ({total_chunks} chunks, {fp_array.shape[0]} segments) to {output_dir}")
        if new_songs_count > 0:
            print(f"  Average chunks per song: {total_chunks / new_songs_count:.1f}")
        return new_songs_count
    elif skipped > 0:
        print(f"✓ No new songs to index (all {skipped} songs already indexed)")
        return 0

    return 0


def main():
    parser = argparse.ArgumentParser(description='Index songs for recognition')
    parser.add_argument('--approach', '-a', choices=['shazam', 'grafp'], required=True)
    parser.add_argument('--folder', '-f', type=str, default='./downloads')
    parser.add_argument('--output', '-o', type=str, default='./fingerprints')
    parser.add_argument('--pattern', '-p', type=str, default='*.flac')
    parser.add_argument('--checkpoint', type=str, default=None, help='GraFP checkpoint')
    parser.add_argument('--config', type=str, default='approaches/grafp/config/grafp.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    
    # Chunking parameters for GraFP
    parser.add_argument('--chunk-duration', type=float, default=60.0,
                       help='Duration of each chunk in seconds (GraFP only, default: 60)')
    parser.add_argument('--chunk-overlap', type=float, default=15.0,
                       help='Overlap between chunks in seconds (GraFP only, default: 15)')
    
    args = parser.parse_args()
    
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    folder = Path(args.folder).expanduser()
    output = Path(args.output).expanduser()
    
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)
    
    # Validate chunking parameters
    if args.chunk_overlap >= args.chunk_duration:
        print(f"Error: Chunk overlap ({args.chunk_overlap}s) must be less than chunk duration ({args.chunk_duration}s)")
        sys.exit(1)
    
    if args.approach == 'shazam':
        index_shazam(folder, output / "shazam", args.pattern)
        
    elif args.approach == 'grafp':
        if not args.checkpoint:
            print("Error: --checkpoint required for GraFP")
            sys.exit(1)
        index_grafp(folder, output / "grafp", args.checkpoint, 
                   args.config, args.device, args.pattern,
                   args.chunk_duration, args.chunk_overlap)


if __name__ == '__main__':
    main()