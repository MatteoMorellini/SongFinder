#!/usr/bin/env python3
"""
Song recognition CLI.

Usage:
    python scripts/recognize.py --approach shazam --query audio.mp3
    python scripts/recognize.py --approach grafp --query audio.flac --checkpoint checkpoints/model.pth
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='SongFinder - Song Recognition')
    parser.add_argument('--approach', '-a', choices=['shazam', 'grafp'], required=True,
                        help='Recognition approach to use')
    parser.add_argument('--query', '-q', type=str, required=True,
                        help='Path to query audio file')
    parser.add_argument('--db-path', type=str, default=None,
                        help='Path to database directory (default: fingerprints/<approach>)')
    parser.add_argument('--clip-length', type=float, default=None,
                        help='Clip length in seconds (for testing with shorter clips)')
    parser.add_argument('--snr', type=float, default=None,
                        help='SNR in dB for noise injection (for testing robustness)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (GraFP only)')

    args = parser.parse_args()
    query_path = Path(args.query)

    if not query_path.exists():
        print(f"Error: Query file not found: {query_path}")
        sys.exit(1)

    if args.approach == 'shazam':
        from approaches.shazam import ShazamRecognizer

        recognizer = ShazamRecognizer()
        db_path = Path(args.db_path) if args.db_path else Path("fingerprints/shazam")
        recognizer.load(db_path)

        print(f"Recognizing: {query_path.name}")
        print(f"Database: {recognizer.num_indexed_songs} songs indexed")

        song_name, score, metadata = recognizer.recognize(
            query_path,
            clip_length_sec=args.clip_length,
            snr_db=args.snr,
            debug=True
        )

        if song_name:
            print(f"\nMatch: {song_name} (score={score}, hashes={metadata.get('num_matched_hashes', 'N/A')})")
        else:
            print("\nNo match found")

    elif args.approach == 'grafp':
        from approaches.grafp.inference import load_model, load_fingerprints, get_or_build_index, recognize as grafp_recognize
        from approaches.grafp.util import load_config
        from approaches.grafp.modules.transformations import AudioTransform
        import torch
        import soundfile as sf
        import torchaudio

        if not args.checkpoint:
            print("Error: --checkpoint required for GraFP")
            sys.exit(1)

        cfg = load_config('approaches/grafp/config/grafp.yaml')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = load_model(cfg, args.checkpoint)
        transform = AudioTransform(cfg).to(device)

        db_path = Path(args.db_path) if args.db_path else Path("fingerprints/grafp")
        try:
            db_fp, db_meta, db_metadata_table = load_fingerprints(db_path)
        except Exception as e:
            print(f"Error loading database from {db_path}: {e}")
            sys.exit(1)

        print(f"Recognizing: {query_path.name}")
        print(f"Database: {len(db_meta)} fingerprints loaded")

        index_path = db_path / "index_ivfpq.faiss"
        index, was_loaded = get_or_build_index(db_fp, str(index_path), use_gpu=True)

        t0 = time.perf_counter()

        signal, sr = sf.read(query_path)
        waveform = torch.from_numpy(signal).float()
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=1)
        if sr != cfg["fs"]:
            waveform = torchaudio.transforms.Resample(sr, cfg["fs"])(waveform)

        segments = transform(waveform.unsqueeze(0).to(device))

        with torch.no_grad():
            _, _, query_fp, _ = model(segments, segments)

        song_name, count = grafp_recognize(query_fp.cpu().numpy(), db_fp, db_meta, index)

        elapsed = time.perf_counter() - t0
        print(f"Recognition time: {elapsed:.3f}s")

        if song_name:
            print(f"\nMatch: {song_name} (votes={count})")
        else:
            print("\nNo match found")


if __name__ == '__main__':
    main()
