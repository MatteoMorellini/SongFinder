# SongFinder

Audio fingerprinting and song identification system implementing two approaches:

- **Shazam**: Traditional signal processing with spectral peaks and constellation hashing
- **GraFP**: Graph Neural Network approach using contrastive learning (ICASSP 2025)

Detailed descriptions of both methods and experimental results are available in the [report](docs/report.pdf)

## Project Structure

```
approaches/
├── shazam/              # Shazam-style implementation
└── grafp/               # GraFP GNN implementation
    ├── encoder/         # Graph encoder and GCN
    └── simclr/          # Contrastive learning

scripts/
├── index_songs.py       # Index songs into database
├── recognize.py         # Query and recognize songs
└── benchmark.py         # Benchmark evaluation

slurm/                   # SLURM job scripts for HPC
```

## Installation

```bash
# Create environment
conda create -n songfinder python=3.12
conda activate songfinder

# Install dependencies
pip install -r requirements.txt

# Install FAISS (GPU or CPU)
conda install -c conda-forge faiss-gpu  # or faiss-cpu
```

## Usage

### Index Songs

```bash
# Shazam approach
python scripts/index_songs.py --approach shazam --folder data/ --pattern "*.flac"

# GraFP approach
python scripts/index_songs.py --approach grafp --folder data/ --pattern "*.flac" \
    --checkpoint checkpoints/model_tc_29_best.pth
```

### Recognize Song

```bash
# Shazam
python scripts/recognize.py --approach shazam --query sample.mp3

# GraFP
python scripts/recognize.py --approach grafp --query sample.mp3 \
    --checkpoint checkpoints/model_tc_29_best.pth
```

### Benchmark

```bash
python scripts/benchmark.py --approach shazam --reference_dir data/ --augmented_dir aug/
```

## API

```python
from approaches.shazam import ShazamRecognizer

recognizer = ShazamRecognizer()
recognizer.load()
song_name, score, metadata = recognizer.recognize("sample.mp3")
```

## References

- Wang, A. (2003). "An Industrial-Strength Audio Search Algorithm"
- Bhattacharjee et al. (2025). "GraFPrint: A GNN-Based Approach for Audio Identification" (ICASSP 2025)
