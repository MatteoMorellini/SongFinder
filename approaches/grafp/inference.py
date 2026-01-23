"""
GraFP inference: fingerprint extraction and similarity search.
"""

import os
import numpy as np
import torch
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not installed. Install with: pip install faiss-cpu")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(cfg, checkpoint_path, k=3):
    """Load a pre-trained GraFP model from checkpoint."""
    from approaches.grafp.encoder.graph_encoder import GraphEncoder
    from approaches.grafp.simclr.simclr import SimCLR
    
    model = SimCLR(cfg, encoder=GraphEncoder(cfg=cfg, in_channels=cfg['n_filters'], k=k))
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model.to(DEVICE))
    else:
        model = model.to(DEVICE)
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint['state_dict']
    
    # Handle DataParallel prefix mismatch
    if torch.cuda.device_count() <= 1 and any('module' in k for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_fingerprints(dataloader, model, transform, output_dir, batch_size=128):
    """Extract fingerprints from audio files and save to disk."""
    fingerprints = []
    metadata = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (audio, meta) in enumerate(dataloader):
        if meta['song'] == '':
            continue
        
        audio = audio.to(DEVICE)
        segments = transform(audio)
        
        for batch in torch.split(segments, batch_size, dim=0):
            with torch.no_grad():
                _, _, z, _ = model(batch, batch)
            
            fingerprints.append(z.cpu().numpy())
            for _ in range(z.shape[0]):
                metadata.append(meta['song'])
        
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(dataloader)}")
    
    fp_array = np.concatenate(fingerprints)
    
    arr = np.memmap(f'{output_dir}/db.mm', dtype='float32', mode='w+', shape=fp_array.shape)
    arr[:] = fp_array[:]
    arr.flush()
    
    np.save(f'{output_dir}/db_shape.npy', fp_array.shape)
    np.save(f'{output_dir}/db_metadata.npy', metadata)
    
    return fp_array.shape[0]


def load_fingerprints(source_dir, name='db'):
    """Load fingerprints from disk."""
    shape = tuple(np.load(f'{source_dir}/{name}_shape.npy'))
    data = np.memmap(f'{source_dir}/{name}.mm', dtype='float32', mode='r', shape=shape)
    
    meta_path = f'{source_dir}/{name}_metadata.npy'
    metadata = np.load(meta_path, allow_pickle=True) if os.path.exists(meta_path) else None
    
    return np.array(data), metadata


def build_index(fingerprints, use_gpu=False):
    """Build FAISS index for similarity search."""
    if not FAISS_AVAILABLE:
        raise ImportError("faiss is required for indexing. Install with: pip install faiss-cpu")
    
    d = fingerprints.shape[1]
    index = faiss.IndexFlatL2(d)
    
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.add(fingerprints.astype('float32'))
    return index


def search(index, query_fingerprints, k=10):
    """Search for similar fingerprints."""
    distances, indices = index.search(query_fingerprints.astype('float32'), k)
    return distances, indices


def recognize(query_fp, db_fingerprints, db_metadata, k=10):
    """Recognize a song from query fingerprints using FAISS."""
    if not FAISS_AVAILABLE:
        return _recognize_numpy(query_fp, db_fingerprints, db_metadata, k)
    
    index = build_index(db_fingerprints, use_gpu=False)
    distances, indices = search(index, query_fp, k)
    
    return _vote_for_song(indices, db_metadata)


def _recognize_numpy(query_fp, db_fingerprints, db_metadata, k=10):
    """Fallback recognition using numpy (slower but no faiss dependency)."""
    from collections import Counter
    
    all_indices = []
    for q in query_fp:
        distances = np.linalg.norm(db_fingerprints - q, axis=1)
        indices = np.argsort(distances)[:k]
        all_indices.append(indices)
    
    return _vote_for_song(np.array(all_indices), db_metadata)


def _vote_for_song(indices, db_metadata):
    """Vote by counting matches per song."""
    from collections import Counter
    
    votes = Counter()
    for idx_row in indices:
        for idx in idx_row:
            if idx >= 0 and idx < len(db_metadata):
                song = db_metadata[idx]
                if isinstance(song, (list, np.ndarray)):
                    song = song[0] if len(song) > 0 else ""
                votes[song] += 1
    
    if votes:
        best_song, count = votes.most_common(1)[0]
        return best_song, count
    return None, 0
