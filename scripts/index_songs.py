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
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

def append_grafp_db(output_dir: Path, new_fp: np.ndarray, new_meta: np.ndarray):
    db_path = output_dir / "db.mm"
    shape_path = output_dir / "db_shape.npy"
    meta_path = output_dir / "db_metadata.npy"

    output_dir.mkdir(parents=True, exist_ok=True)

    if db_path.exists() and shape_path.exists() and meta_path.exists():
        old_shape = tuple(np.load(shape_path))
        old_meta = np.load(meta_path, allow_pickle=True)
        old_db = np.memmap(db_path, dtype="float32", mode="r", shape=old_shape)

        # build combined
        combined_shape = (old_shape[0] + new_fp.shape[0], old_shape[1])
        combined_meta = np.concatenate([old_meta, new_meta])

        # rewrite db.mm with combined content
        new_db = np.memmap(db_path, dtype="float32", mode="w+", shape=combined_shape)
        new_db[:old_shape[0]] = old_db[:]
        new_db[old_shape[0]:] = new_fp
        new_db.flush()

        np.save(shape_path, np.array(combined_shape))
        np.save(meta_path, combined_meta)
    else:
        # first time
        new_db = np.memmap(db_path, dtype="float32", mode="w+", shape=new_fp.shape)
        new_db[:] = new_fp
        new_db.flush()
        np.save(shape_path, np.array(new_fp.shape))
        np.save(meta_path, new_meta)


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
        
        new_meta = np.array(metadata)
        append_grafp_db(output_dir, fp_array, new_meta)

        print(f"✓ Saved {len(audio_files)} songs ({fp_array.shape[0]} segments) to {output_dir}")
        return len(audio_files)
    
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
