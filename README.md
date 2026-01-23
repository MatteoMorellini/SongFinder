# SongFinder

A song recognition system implementing two distinct approaches for audio fingerprinting and identification:

1. **Shazam-style**: Traditional signal processing with spectral peaks and constellation hashing
2. **GraFP**: Graph Neural Network based approach using contrastive learning (SimCLR)

## Project Structure

```
SongFinder/
├── approaches/
│   ├── base.py              # Abstract base class for recognizers
│   ├── shazam/              # Shazam-style implementation
│   │   ├── audio.py         # Audio loading, spectrogram, peak finding
│   │   ├── config.py        # Configuration constants
│   │   ├── hashing.py       # Constellation hashing
│   │   ├── recognizer.py    # Main ShazamRecognizer class
│   │   └── ...
│   └── grafp/               # GraFP GNN implementation
│       ├── encoder/         # Graph encoder and GCN library
│       ├── simclr/          # Contrastive learning
│       ├── train.py         # Training script
│       ├── generate.py      # Fingerprint generation
│       └── ...
├── scripts/
│   ├── recognize.py         # Unified recognition CLI
│   └── index_songs.py       # Song indexing CLI
└── data/                    # Audio files (gitignored)
```

## Quick Start

### Shazam Approach

1. **Index songs:**
   ```bash
   python scripts/index_songs.py --approach shazam --folder data/ --pattern "*.flac"
   ```

2. **Recognize a song:**
   ```bash
   python scripts/recognize.py --approach shazam --query test/sample.mp3
   ```

### GraFP Approach

1. **Train the model:**
   ```bash
   python approaches/grafp/train.py --config approaches/grafp/config/grafp.yaml \
       --train_dir /path/to/training/data --val_dir /path/to/validation/data
   ```

2. **Generate fingerprints:**
   ```bash
   python approaches/grafp/generate.py --test_dir /path/to/audio \
       --ckp /path/to/checkpoint.pth --output_dir output/
   ```

3. **Query:**
   ```bash
   python approaches/grafp/query.py --test_dir /path/to/query.json
   ```

## Both Approaches Use a Common Interface

```python
from approaches.shazam import ShazamRecognizer

recognizer = ShazamRecognizer()
recognizer.load()
song_name, score, metadata = recognizer.recognize(query_path)
```

## Requirements

**Shazam approach:**
- librosa
- numpy
- soundfile
- tqdm

**GraFP approach:**
- torch, torchaudio
- timm
- faiss-gpu (or faiss-cpu)
- See `approaches/grafp/requirements.txt`

## References

- Shazam: Wang, A. (2003). "An Industrial-Strength Audio Search Algorithm"
- GraFP: Bhattacharjee et al. (2025). "GraFPrint: A GNN-Based Approach for Audio Identification" (ICASSP 2025)
