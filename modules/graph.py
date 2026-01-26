from pyexpat import model
import torch
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader
from pathlib import Path
import librosa
import os
from tqdm import tqdm
from .config import BANDS, N_FFT, PLOT_SPECTROGRAM, TARGET_SR, HOP_LENGTH
from .db import load_db, save_db, get_song_id
from .audio import load_audio, extract_spectrogram, find_peaks, plot_spectrogram_and_save
from .hashing import build_hashes, add_hashes_to_table
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(3, hidden_channels) # input features are 3: (x, y, intensity) 
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x


def neighbors_to_edge_index(neighbors, undirected=False, num_nodes=None):
    """
    Convert a neighbor list representation to PyTorch Geometric edge index format.
    
    Args:
        neighbors: List of neighbor lists where neighbors[i] contains indices of nodes adjacent to node i
        undirected: If True, adds reverse edges to make the graph undirected (doubling all connections)
        num_nodes: (unused) Number of nodes in the graph
    
    Returns:
        torch.Tensor: Edge index in PyG format with shape [2, num_edges], where first row contains source nodes
                     and second row contains destination nodes
    """
    # Initialize lists to store source and destination node indices
    src, dst = [], []
    
    # Iterate through each node and its neighbors
    for i, nbrs in enumerate(neighbors):
        # For each neighbor of node i, add an edge from i to that neighbor
        for j in nbrs:
            src.append(i)
            dst.append(j)

    # Convert to PyTorch tensor with long dtype (required by PyG)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # If undirected, add reverse edges so all connections are bidirectional
    if undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    return edge_index

def build_graph(nodes, edges, label):
    # N nodes, each has 3 features: (x, y, intensity)
    # nodes: (N, 3)

    # edges: for each node i you have 5 neighbors (N, 5) with node indices
    # edges[i] = [j1, j2, j3, j4, j5]
    # ---- Build PyG tensors ----
    nodes = torch.as_tensor(nodes, dtype=torch.float)          # [N, 3]
    edge_index = neighbors_to_edge_index(edges, undirected=True)

    data = Data(x=nodes, edge_index=edge_index)
    data.y = torch.tensor([label], dtype=torch.long)

    return data

def build_song_graph(audio_path: Path):

    #print(f"Processing {song_name}")
    signal, sr = load_audio(audio_path)
    spectrogram = extract_spectrogram(signal, sr)
    peaks = find_peaks(spectrogram, BANDS)

    freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=N_FFT)

    fingerprints, edges = build_hashes(peaks, freqs, song_id=None)

    label = 0 if 'Bon Iver' in audio_path.name else 1

    graph = build_graph(peaks, edges, label)

    ei = graph.edge_index
    N = graph.num_nodes

    # Sanity checks
    assert ei.shape[0] == 2
    assert ei.dtype == torch.long
    assert ei.numel() > 0
    assert ei.min().item() >= 0
    assert ei.max().item() < N

    torch.manual_seed(12345)

    partitioning = False

    if not partitioning: return graph
    
    try:
        cluster_data = ClusterData(graph, num_parts=8, recursive=False)
        train_loader = ClusterLoader(cluster_data, batch_size=4, shuffle=True)  
    except Exception as e:
        print("ClusterData failed, falling back to full-graph:", e)
        cluster_data = graph
        train_loader = ClusterLoader(cluster_data, batch_size=4, shuffle=True)  
    
    total_num_nodes = 0
    for step, sub_data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
        print(sub_data)
        print()
        total_num_nodes += sub_data.num_nodes

    print(f'Iterated over {total_num_nodes} of {graph.num_nodes} nodes!')

    # todo: return partitioned graph


def train_graph(
    folder: Path,
    pattern: str = "*.flac",
):
    #audio_paths = list(folder.glob(pattern))
    audio_paths = [
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Bon Iver - Bon Iver, Bon Iver - 01-01 Perth.flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Bon Iver - Bon Iver, Bon Iver - 01-02 Minnesota, WI.flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Bon Iver - Bon Iver, Bon Iver - 01-03 Holocene.flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Bon Iver - Bon Iver, Bon Iver - 01-04 Towers.flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Bon Iver - Bon Iver, Bon Iver - 01-05 Michicant.flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Bon Iver - Bon Iver, Bon Iver - 01-06 Hinnom, TX.flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Bon Iver - Bon Iver, Bon Iver - 01-07 Wash..flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Bon Iver - Bon Iver, Bon Iver - 01-08 Calgary.flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Kendrick Lamar - DAMN. - 01-01 BLOOD..flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Kendrick Lamar - DAMN. - 01-02 DNA..flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Kendrick Lamar - DAMN. - 01-03 YAH..flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Kendrick Lamar - DAMN. - 01-04 ELEMENT..flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Kendrick Lamar - DAMN. - 01-05 FEEL..flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Kendrick Lamar - DAMN. - 01-06 LOYALTY..flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Kendrick Lamar - DAMN. - 01-07 PRIDE..flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Kendrick Lamar - DAMN. - 01-08 HUMBLE..flac',
    ]

    test_audio_paths = [
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Bon Iver - Bon Iver, Bon Iver - 01-09 Lisbon, OH.flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Bon Iver - Bon Iver, Bon Iver - 01-10 Beth_Rest.flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Bon Iver - For Emma, Forever Ago - 01-01 Flume.flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Bon Iver - For Emma, Forever Ago - 01-02 Lump Sum.flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Kendrick Lamar - DAMN. - 01-09 LUST..flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Kendrick Lamar - DAMN. - 01-10 LOVE..flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Kendrick Lamar - DAMN. - 01-11 XXX..flac',
        '/Users/matteomorellini/Desktop/code/shazam_clone/data/Kendrick Lamar - DAMN. - 01-12 FEAR..flac',
    ]
    audio_paths = [Path(p) for p in audio_paths]
    test_audio_paths = [Path(p) for p in test_audio_paths]
    model = GCN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    print(model)
    
    graphs = []
    test_graphs = []

    for audio_path in audio_paths:
        print(f"Building graph for {audio_path.name}")
        graphs.append(build_song_graph(audio_path))
        
    for audio_path in test_audio_paths:
        print(f"Building graph for {audio_path.name}")
        test_graphs.append(build_song_graph(audio_path))

    loader = DataLoader(graphs, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=10, shuffle=False)

    for epoch in range(1, 171):
        model.train()

        for data in tqdm(loader): 
            out = model(data.x, data.edge_index, data.batch)  
            loss = criterion(out, data.y)  
            loss.backward() 
            optimizer.step() 
            optimizer.zero_grad()  
        
        model.eval()

        correct = 0
        for data in loader:  
            out = model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1) 
            correct += int((pred == data.y).sum()) 
        train_acc = correct / len(loader.dataset) 

        correct = 0
        for data in test_loader: 
            out = model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)  
            correct += int((pred == data.y).sum())  
        test_acc = correct / len(test_loader.dataset) 

        #train_acc = test(train_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
