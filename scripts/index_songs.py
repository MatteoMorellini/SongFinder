#!/usr/bin/env python3
"""
Index songs into the fingerprint database for Shazam or GraFP.

Usage:
    python scripts/index_songs.py --approach shazam --folder ~/datasets/fma_small
    python scripts/index_songs.py --approach grafp --folder ~/datasets/fma_small \
                                  --checkpoint path/to/model.pth
"""

import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def append_grafp_db(output_dir: Path, new_fp: np.ndarray, new_meta: np.ndarray, new_metadata_dict: dict = None):
    """Append new fingerprints to an existing (or new) GraFP database on disk."""
    import pickle

    db_path = output_dir / "db.mm"
    shape_path = output_dir / "db_shape.npy"
    meta_path = output_dir / "db_metadata.npy"
    metadata_table_path = output_dir / "metadata_table.pkl"

    output_dir.mkdir(parents=True, exist_ok=True)

    if metadata_table_path.exists():
        with open(metadata_table_path, 'rb') as f:
            metadata_table = pickle.load(f)
    else:
        metadata_table = {}

    if new_metadata_dict:
        metadata_table.update(new_metadata_dict)

    if db_path.exists() and shape_path.exists() and meta_path.exists():
        old_shape = tuple(np.load(shape_path))
        old_meta = np.load(meta_path, allow_pickle=True)
        old_db = np.memmap(db_path, dtype="float32", mode="r", shape=old_shape)

        old_meta = np.asarray(old_meta, dtype=object).reshape(-1)
        new_meta = np.asarray(new_meta, dtype=object).reshape(-1)

        if old_meta.shape[0] != old_shape[0]:
            raise ValueError(f"Metadata length ({old_meta.shape[0]}) != db rows ({old_shape[0]})")
        if new_meta.shape[0] != new_fp.shape[0]:
            raise ValueError(f"New metadata length ({new_meta.shape[0]}) != new_fp rows ({new_fp.shape[0]})")

        combined_shape = (old_shape[0] + new_fp.shape[0], old_shape[1])
        combined_meta = np.concatenate([old_meta, new_meta])

        # Copy old data to RAM before truncating the memmap file
        old_data_copy = np.array(old_db)
        del old_db

        new_db = np.memmap(db_path, dtype="float32", mode="w+", shape=combined_shape)
        new_db[:old_shape[0]] = old_data_copy
        new_db[old_shape[0]:] = new_fp
        new_db.flush()
        del new_db

        np.save(shape_path, np.array(combined_shape))
        np.save(meta_path, combined_meta)

    else:
        new_db = np.memmap(db_path, dtype="float32", mode="w+", shape=new_fp.shape)
        new_db[:] = new_fp
        new_db.flush()
        del new_db
        np.save(shape_path, np.array(new_fp.shape))
        np.save(meta_path, new_meta)

    with open(metadata_table_path, 'wb') as f:
        pickle.dump(metadata_table, f)


def index_shazam(folder: Path, output_dir: Path, pattern: str):
    """Index songs using Shazam approach."""
    from approaches.shazam import ShazamRecognizer
    from tqdm import tqdm

    print("\n=== Shazam Indexing ===")

    recognizer = ShazamRecognizer()

    try:
        recognizer.load(output_dir)
        print(f"Loaded existing: {recognizer.num_indexed_songs} songs")
    except Exception:
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

    print(f"Saved {recognizer.num_indexed_songs} songs to {output_dir}")
    return recognizer.num_indexed_songs


def chunk_audio(waveform: np.ndarray, sr: int, chunk_duration: float,
                overlap_duration: float):
    """Split audio into overlapping chunks, discarding chunks < 50% target duration."""
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap_duration * sr)
    hop_samples = chunk_samples - overlap_samples

    chunks = []
    start_sample = 0

    while start_sample < len(waveform):
        end_sample = min(start_sample + chunk_samples, len(waveform))
        chunk = waveform[start_sample:end_sample]

        if len(chunk) >= int(chunk_duration * 0.5 * sr):
            start_time = start_sample / sr
            chunks.append((chunk, start_time))

        if end_sample >= len(waveform):
            break

        start_sample += hop_samples

    return chunks


def index_grafp(folder: Path, output_dir: Path, checkpoint: str,
                config: str, device: str, pattern: str,
                chunk_duration: float = 60.0, chunk_overlap: float = 15.0):
    """Index songs using GraFP approach with chunking."""
    import torch
    import pickle
    from tqdm import tqdm
    from approaches.grafp import load_config, load_model
    from approaches.grafp.modules.transformations import AudioTransform
    from utils.augmentation import load_audio
    from utils.metadata import extract_metadata

    print("\n=== GraFP Indexing ===")
    print(f"Chunk: {chunk_duration}s duration, {chunk_overlap}s overlap")

    cfg = load_config(config)
    model = load_model(cfg, checkpoint)
    transform = AudioTransform(cfg).to(device)

    metadata_table_path = output_dir / "metadata_table.pkl"
    if metadata_table_path.exists():
        with open(metadata_table_path, 'rb') as f:
            existing_metadata = pickle.load(f)
        print(f"Existing metadata: {len(existing_metadata)} songs indexed")
    else:
        existing_metadata = {}

    audio_files = list(folder.rglob(pattern))
    if not audio_files:
        audio_files = list(folder.rglob("*.mp3"))

    print(f"Found {len(audio_files)} audio files")

    fingerprints = []
    metadata = []
    new_metadata_dict = {}

    model.eval()
    total_chunks = 0
    skipped = 0

    for f in tqdm(audio_files, desc="Processing songs"):
        filename = f.stem

        if filename in existing_metadata:
            skipped += 1
            continue

        song_metadata = extract_metadata(f)
        new_metadata_dict[filename] = song_metadata

        try:
            waveform_np, sr = load_audio(f, cfg['fs'])
            chunks = chunk_audio(waveform_np, sr, chunk_duration, chunk_overlap)

            if not chunks:
                print(f"Warning: No valid chunks for {f.name}")
                continue

            for chunk_idx, (chunk_waveform, start_time) in enumerate(chunks):
                try:
                    waveform = torch.from_numpy(chunk_waveform).float()
                    segments = transform(waveform.unsqueeze(0).to(device))

                    with torch.no_grad():
                        _, _, z, _ = model(segments, segments)

                    fingerprints.append(z.cpu().numpy())

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
        print(f"Saved {new_songs_count} new songs ({total_chunks} chunks, {fp_array.shape[0]} segments) to {output_dir}")
        return new_songs_count
    elif skipped > 0:
        print(f"No new songs to index (all {skipped} songs already indexed)")
        return 0

    return 0


def main():
    parser = argparse.ArgumentParser(description='Index songs for recognition')
    parser.add_argument('--approach', '-a', choices=['shazam', 'grafp'], required=True,
                        help='Recognition approach to use')
    parser.add_argument('--folder', '-f', type=str, default='./downloads',
                        help='Folder containing audio files to index')
    parser.add_argument('--output', '-o', type=str, default='./fingerprints',
                        help='Output directory for fingerprint database')
    parser.add_argument('--pattern', '-p', type=str, default='*.flac',
                        help='Glob pattern for audio files')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (GraFP only)')
    parser.add_argument('--config', type=str, default='approaches/grafp/config/grafp.yaml',
                        help='Path to GraFP config YAML')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference (cuda/cpu)')
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

    if args.chunk_overlap >= args.chunk_duration:
        print(f"Error: Chunk overlap ({args.chunk_overlap}s) must be less than duration ({args.chunk_duration}s)")
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
