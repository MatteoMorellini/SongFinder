"""
GraFP utility functions for inference.
"""

import os
import json
import glob
import yaml
import torch
import numpy as np


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_index(cfg, data_dir, ext=['wav', 'mp3'], mode="inference"):
    """Load or create an index of audio files."""
    if data_dir.endswith('.json'):
        with open(data_dir, 'r') as f:
            return json.load(f)
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    fpaths = [p for p in glob.glob(os.path.join(data_dir, '**/*.*'), recursive=True)
              if p.split('.')[-1] in ext]
    
    indices = list(range(len(fpaths)))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    return {str(i): fpaths[ix] for i, ix in enumerate(indices)}


def get_frames(y, frame_length, hop_length):
    return y.unfold(0, size=frame_length, step=hop_length)


def qtile_normalize(y, q, eps=1e-8):
    return y / (eps + torch.quantile(y.abs(), q=q))


def qtile_norm(y, q, eps=1e-8):
    return eps + torch.quantile(y.abs(), q=q)


def query_len_from_seconds(seconds, overlap, dur):
    hop = dur * (1 - overlap)
    return int((seconds - dur) / hop + 1)


def seconds_from_query_len(query_len, overlap, dur):
    hop = dur * (1 - overlap)
    return int((query_len - 1) * hop + dur)
