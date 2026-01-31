#!/usr/bin/env python3
"""
Unified song recognition CLI.

Usage:
    python scripts/recognize.py --approach shazam --query audio.mp3
    python scripts/recognize.py --approach shazam --query audio.mp3 --clip-length 10 --snr 5
"""

import argparse
from pathlib import Path
import sys
import torchaudio
import soundfile as sf
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='SongFinder - Unified Song Recognition')
    parser.add_argument('--approach', '-a', choices=['shazam', 'grafp'], required=True,
                        help='Recognition approach to use')
    parser.add_argument('--query', '-q', type=str, required=True,
                        help='Path to query audio file')
    parser.add_argument('--db-path', type=str, default=None,
                        help='Path to database directory (default: project root)')
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
            debug = True
        )
        
        if song_name:
            print(f"\n✓ Match found: {song_name}")
            print(f"  Score: {score}")
            print(f"  Matched hashes: {metadata.get('num_matched_hashes', 'N/A')}")
        else:
            print("\n✗ No match found")
            
    elif args.approach == 'grafp':
        from approaches.grafp.inference import load_model, load_fingerprints, build_index, recognize as grafp_recognize
        from approaches.grafp.util import load_config
        from approaches.grafp.modules.transformations import AudioTransform
        import torch
        
        cfg_path = 'approaches/grafp/config/grafp.yaml'
        cfg = load_config(cfg_path)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not args.checkpoint:
            print("Error: --checkpoint required for GraFP")
            sys.exit(1)
            
        model = load_model(cfg, args.checkpoint)
        transform = AudioTransform(cfg).to(device)
        
        # Load database fingerprints
        db_path = Path(args.db_path) if args.db_path else Path("fingerprints/grafp")
        try:
            db_fp, db_meta = load_fingerprints(db_path)
        except Exception as e:
            print(f"Error loading database from {db_path}: {e}")
            sys.exit(1)
            
        print(f"Recognizing: {query_path.name}")
        print(f"Database: {len(db_meta)} fingerprints loaded")

        index = build_index(db_fp, use_gpu=False)
        
        timings = {}

        t_total0 = time.perf_counter()

        # 1) Load audio
        t0 = time.perf_counter()
        signal, sr = sf.read(query_path)
        timings["load_audio_sf"] = time.perf_counter() - t0
        print(f"Load audio (soundfile): {timings['load_audio_sf']:.4f}s  (sr={sr}, shape={getattr(signal,'shape',None)})")

        # 2) To torch float tensor
        t0 = time.perf_counter()
        waveform = torch.from_numpy(signal).float()
        timings["to_torch_float"] = time.perf_counter() - t0
        print(f"To torch float: {timings['to_torch_float']:.4f}s  (dtype={waveform.dtype}, shape={tuple(waveform.shape)})")

        # 3) Mono conversion if needed
        t0 = time.perf_counter()
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=1)  # (num_samples,)
        timings["to_mono"] = time.perf_counter() - t0
        print(f"To mono: {timings['to_mono']:.4f}s  (shape={tuple(waveform.shape)})")

        # 4) Resample if needed
        t0 = time.perf_counter()
        if sr != cfg["fs"]:
            # Creating the Resample module each call costs time; see note below.
            waveform = torchaudio.transforms.Resample(sr, cfg["fs"])(waveform)
        timings["resample"] = time.perf_counter() - t0
        print(f"Resample: {timings['resample']:.4f}s  (target_fs={cfg['fs']}, shape={tuple(waveform.shape)})")

        # 5) Segment/transform (often CPU-heavy)
        t0 = time.perf_counter()
        # If transform expects batch dimension: (1, T)
        segments = transform(waveform.unsqueeze(0).to(device))
        # If using CUDA, sync so timing reflects actual GPU work
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings["transform_segments"] = time.perf_counter() - t0
        print(f"Transform/segment: {timings['transform_segments']:.4f}s  (segments_shape={tuple(segments.shape)})")

        # 6) Model forward (GPU timing needs sync)
        t0 = time.perf_counter()
        with torch.no_grad():
            _, _, query_fp, _ = model(segments, segments)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings["model_forward"] = time.perf_counter() - t0
        print(f"Model forward: {timings['model_forward']:.4f}s  (query_fp_shape={tuple(query_fp.shape)})")

        # 7) Move fingerprint to CPU + numpy
        t0 = time.perf_counter()
        query_fp_np = query_fp.detach().cpu().numpy()
        timings["fp_to_cpu_numpy"] = time.perf_counter() - t0
        print(f"FP to CPU numpy: {timings['fp_to_cpu_numpy']:.4f}s  (dtype={query_fp_np.dtype}, shape={query_fp_np.shape})")

        # 8) DB recognition
        song_name, count = grafp_recognize(query_fp_np, db_fp, db_meta, index)

        timings["total"] = time.perf_counter() - t_total0
        print(f"Total recognition time: {timings['total']:.4f}s")
        
        if song_name:
            print(f"\n✓ Match found: {song_name}")
            print(f"  Confidence (votes): {count}")
        else:
            print("\n✗ No match found")
        

if __name__ == '__main__':
    main()
