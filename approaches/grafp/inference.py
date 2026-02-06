"""
GraFP inference: fingerprint extraction and similarity search.
"""

from collections import Counter

import os
import numpy as np
import torch
from pathlib import Path
import time

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
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['state_dict']
    
    if torch.cuda.device_count() <= 1 and any('module' in k for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Remap legacy checkpoint keys (convs -> conv)
    state_dict = {k.replace('peak_extractor.convs.', 'peak_extractor.conv.'): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
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
    """
    Load fingerprints from disk.

    Returns:
        tuple: (data, metadata, metadata_table)
            - data: fingerprints array
            - metadata: array of filenames (one per segment)
            - metadata_table: dict mapping filename -> {title, artist, album, filename}
    """
    import pickle

    shape = tuple(np.load(f'{source_dir}/{name}_shape.npy'))
    data = np.memmap(f'{source_dir}/{name}.mm', dtype='float32', mode='r', shape=shape)

    meta_path = f'{source_dir}/{name}_metadata.npy'
    metadata = np.load(meta_path, allow_pickle=True) if os.path.exists(meta_path) else None

    metadata_table_path = os.path.join(source_dir, 'metadata_table.pkl')
    if os.path.exists(metadata_table_path):
        with open(metadata_table_path, 'rb') as f:
            metadata_table = pickle.load(f)
    else:
        metadata_table = {}

    return np.array(data), metadata, metadata_table


def get_index(index_type,
              train_data,
              train_data_shape,
              use_gpu=True,
              max_nitem_train=2e7,
              n_centroids=64,
):
    """
    • Create FAISS index
    • Train index using (partial) data
    • Return index
    Parameters
    ----------
    index_type : (str)
        Index type must be one of {'L2', 'IVF', 'IVFPQ', 'IVFPQ-RR',
                                   'IVFPQ-ONDISK', HNSW'}
    train_data : (float32)
        numpy.memmap or numpy.ndarray
    train_data_shape : list(int, int)
        Data shape (n, d). n is the number of items. d is dimension.
    use_gpu: (bool)
        If False, use CPU. Default is True.
    max_nitem_train : (int)
        Max number of items to be used for training index. Default is 1e7.
    Returns
    -------
    index : (faiss.swigfaiss_avx2.GpuIndex***)
        Trained FAISS index.
    References:
        https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    """
    if use_gpu:
        GPU_RESOURCES = faiss.StandardGpuResources()
        GPU_OPTIONS = faiss.GpuClonerOptions()
        GPU_OPTIONS.useFloat16 = True

    d = train_data_shape[1]
    index = faiss.IndexFlatL2(d)

    mode = index_type.lower()
    print(f'Creating index: {mode}')
    if mode == 'l2':
        pass
    elif mode == 'ivf':
        nlist = 400
        index = faiss.IndexIVFFlat(index, d, nlist)
    elif mode == 'ivfpq':
        code_sz = 8
        nbits = 8
        index = faiss.IndexIVFPQ(index, d, n_centroids, code_sz, nbits)

    elif mode == 'lsh':
        nbits = 256
        index = faiss.IndexLSH(d, nbits)


    elif mode == 'ivfpq-rr':
        code_sz = 64
        nbits = 8
        M_refine = 4
        nbits_refine = 4
        index = faiss.IndexIVFPQR(index, d, n_centroids, code_sz, nbits,
                                  M_refine, nbits_refine)
    elif mode == 'ivfpq-ondisk':
        if use_gpu:
            raise NotImplementedError(f'{mode} is only available in CPU.')
        raise NotImplementedError(mode)
    elif mode == 'hnsw':
        if use_gpu:
            raise NotImplementedError(f'{mode} is only available in CPU.')
        else:
            M = 16
            index = faiss.IndexHNSWFlat(d, M)
            index.hnsw.efConstruction = 80
            index.verbose = True
            index.hnsw.search_bounded_queue = True
    else:
        raise ValueError(mode.lower())

    if use_gpu:
        print('Copy index to GPU.')
        index = faiss.index_cpu_to_gpu(GPU_RESOURCES, 0, index, GPU_OPTIONS)

    start_time = time.time()
    if len(train_data) > max_nitem_train:
        print('Training index using {:>3.2f} % of data...'.format(
            100. * max_nitem_train / len(train_data)))
        sel_tr_idx = np.random.permutation(len(train_data))
        sel_tr_idx = sel_tr_idx[:int(max_nitem_train)]
        index.train(train_data[sel_tr_idx,:])
    else:
        print('Training index...')
        index.train(train_data)
    print(f'Training completed in {time.time() - start_time:.2f}s')

    index.nprobe = 20
    return index


def build_index(fingerprints, use_gpu=True):
    """Build FAISS index for similarity search using IVFPQ."""
    if not FAISS_AVAILABLE:
        raise ImportError("faiss is required for indexing. Install with: pip install faiss-cpu")
    
    max_train = 1e7
    n_centroids = 256
    index = get_index('ivfpq', fingerprints, fingerprints.shape, use_gpu,
                      max_train, n_centroids=n_centroids)
    
    index.add(fingerprints.astype('float32'))
    return index


def save_index(index, path, use_gpu=False):
    """
    Save FAISS index to disk.
    
    Parameters
    ----------
    index : faiss.Index
        The FAISS index to save
    path : str
        Path to save the index (e.g., 'fingerprints/grafp/index.faiss')
    use_gpu : bool
        If True, the index is on GPU and needs to be copied to CPU first
    """
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    if use_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
        print(f'Copying index from GPU to CPU for saving...')
        index_cpu = faiss.index_gpu_to_cpu(index)
    else:
        index_cpu = index
    
    faiss.write_index(index_cpu, path)
    print(f'Index saved to: {path}')


def load_index(path, use_gpu=False):
    """
    Load FAISS index from disk.
    
    Parameters
    ----------
    path : str
        Path to the saved index file
    use_gpu : bool
        If True, copy the loaded index to GPU
        
    Returns
    -------
    index : faiss.Index
        The loaded FAISS index
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found: {path}")
    
    print(f'Loading index from: {path}')
    index = faiss.read_index(path)
    
    if use_gpu and hasattr(faiss, 'StandardGpuResources'):
        print('Copying index to GPU...')
        res = faiss.StandardGpuResources()
        opts = faiss.GpuClonerOptions()
        opts.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, opts)
    
    if hasattr(index, 'nprobe'):
        index.nprobe = 20
    
    return index


def get_or_build_index(fingerprints, index_path, use_gpu=True, force_rebuild=False):
    """
    Load index from disk if it exists, otherwise build and save it.
    
    Parameters
    ----------
    fingerprints : np.ndarray
        Database fingerprints (only used if building)
    index_path : str
        Path to the index file
    use_gpu : bool
        Whether to use GPU for the index
    force_rebuild : bool
        If True, rebuild even if the file exists
        
    Returns
    -------
    index : faiss.Index
        The FAISS index (loaded or newly built)
    was_loaded : bool
        True if the index was loaded from disk, False if built
    """
    import os
    
    if os.path.exists(index_path) and not force_rebuild:
        print(f'Found existing index at {index_path}')
        index = load_index(index_path, use_gpu=use_gpu)
        return index, True
    else:
        print(f'Building new index...')
        index = build_index(fingerprints, use_gpu=use_gpu)
        save_index(index, index_path, use_gpu=use_gpu)
        return index, False


def search(index, query_fingerprints, k=10):
    """Search for similar fingerprints."""
    distances, indices = index.search(query_fingerprints.astype('float32'), k)
    return distances, indices


def recognize(query_fp, db_fingerprints, db_metadata, index, k=10, top_songs_entropy=10):

    """Recognize a song from query fingerprints using FAISS."""
    if not FAISS_AVAILABLE:
        return _recognize_numpy(query_fp, db_fingerprints, db_metadata, k)

    distances, indices = search(index, query_fp, k)
    
    return _vote_for_song(indices, db_metadata, top_songs_entropy)


def _recognize_numpy(query_fp, db_fingerprints, db_metadata, k=10, top_songs_entropy=10):
    """Fallback recognition using numpy (slower but no faiss dependency)."""
    from collections import Counter
    
    all_indices = []
    for q in query_fp:
        distances = np.linalg.norm(db_fingerprints - q, axis=1)
        indices = np.argsort(distances)[:k]
        all_indices.append(indices)
    
    return _vote_for_song(np.array(all_indices), db_metadata, top_songs_entropy)


def _vote_for_song(indices, db_metadata, top_songs_entropy):
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
    print('most votes:', votes.most_common(10))
    if votes:
        best_song, best_count = votes.most_common(1)[0]
        # restrict to top-K to avoid long tails dominating entropy

        items = votes.most_common(top_songs_entropy) if top_songs_entropy else list(votes.items()) 
        # None to consider the whole domain in entropy metric
        counts = np.array([c for _, c in items], dtype=np.float64)

        sum_counts = counts.sum()
        if sum_counts <= 0 or len(counts) == 0:
            confidence = 0.0
        else:
            p = counts / sum_counts
            eps = 1e-12  # avoid log(0)
            H = -np.sum(p * np.log(p + eps))  # entropy
            H_norm = H / np.log(len(p)) if len(p) > 1 else 0.0  # normalize to [0,1]
            confidence = float(1.0 - H_norm)
            confidence = max(0.0, min(1.0, confidence))  # clamp for safety

        return best_song, confidence
        
    return None, 0.0