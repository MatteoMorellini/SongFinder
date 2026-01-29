#!/usr/bin/env python3
"""
Index songs into the database for Shazam or GraFP.

Usage:
    # Shazam
    python scripts/index_songs.py --approach shazam --folder ~/datasets/fma_small
    
    # GraFP (requires checkpoint)
    python scripts/index_songs.py --approach grafp --folder ~/datasets/fma_small \
                                  --checkpoint path/to/model.pth
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


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


def index_grafp(folder: Path, output_dir: Path, checkpoint: str, 
                config: str, device: str, pattern: str):
    """Index songs using GraFP approach."""
    import torch
    import torchaudio
    import numpy as np
    from tqdm import tqdm
    from approaches.grafp import load_config, load_model
    from approaches.grafp.modules.transformations import AudioTransform
    
    print("\n=== GraFP Indexing ===")
    
    cfg = load_config(config)
    model = load_model(cfg, checkpoint)
    transform = AudioTransform(cfg).to(device)
    
    audio_files = list(folder.rglob(pattern))
    if not audio_files:
        audio_files = list(folder.rglob("*.mp3"))
    
    print(f"Found {len(audio_files)} audio files")
    
    fingerprints = []
    metadata = []
    
    import soundfile as sf
    
    model.eval()
    for f in tqdm(audio_files, desc="Generating fingerprints"):
        try:
            # Using soundfile instead of torchaudio for better backend stability
            signal, sr = sf.read(f)
            waveform = torch.from_numpy(signal).float()
            
            # Convert to mono if stereo
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=1)
            
            if sr != cfg['fs']:
                waveform = torchaudio.transforms.Resample(sr, cfg['fs'])(waveform)
            
            segments = transform(waveform.unsqueeze(0).to(device))
            
            with torch.no_grad():
                _, _, z, _ = model(segments, segments)
            
            fingerprints.append(z.cpu().numpy())
            for _ in range(z.shape[0]):
                metadata.append(f.stem)
                
        except Exception as e:
            print(f"Error {f.name}: {e}")
    
    if fingerprints:
        fp_array = np.concatenate(fingerprints).astype('float32')
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in the format expected by load_fingerprints (inference.py)
        # Using memmap for better scalability
        arr = np.memmap(output_dir / "db.mm", dtype='float32', mode='w+', shape=fp_array.shape)
        arr[:] = fp_array[:]
        arr.flush()
        
        np.save(output_dir / "db_shape.npy", fp_array.shape)
        np.save(output_dir / "db_metadata.npy", np.array(metadata))
        
        print(f"✓ Saved {len(audio_files)} songs ({fp_array.shape[0]} segments) to {output_dir}")
        return len(audio_files)
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='Index songs for recognition')
    parser.add_argument('--approach', '-a', choices=['shazam', 'grafp'], required=True)
    parser.add_argument('--folder', '-f', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, default='./fingerprints')
    parser.add_argument('--pattern', '-p', type=str, default='*.flac')
    parser.add_argument('--checkpoint', type=str, default=None, help='GraFP checkpoint')
    parser.add_argument('--config', type=str, default='approaches/grafp/config/grafp.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    folder = Path(args.folder).expanduser()
    output = Path(args.output).expanduser()
    
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)
    
    if args.approach == 'shazam':
        index_shazam(folder, output / "shazam", args.pattern)
        
    elif args.approach == 'grafp':
        if not args.checkpoint:
            print("Error: --checkpoint required for GraFP")
            sys.exit(1)
        index_grafp(folder, output / "grafp", args.checkpoint, 
                   args.config, args.device, args.pattern)


if __name__ == '__main__':
    main()
