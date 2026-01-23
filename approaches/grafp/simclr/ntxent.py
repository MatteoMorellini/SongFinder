"""
NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning.
"""

import torch
import torch.nn.functional as F


def ntxent_loss(z_i, z_j, cfg):
    """
    Compute NT-Xent loss between positive pairs.
    
    Args:
        z_i: Embeddings of original samples (batch_size x emb_size)
        z_j: Embeddings of augmented samples (batch_size x emb_size)
        cfg: Config dict with 'tau' temperature parameter
    
    Returns:
        Scalar loss value
    """
    tau = cfg['tau']
    batch_size = z_i.shape[0]
    
    # Stack pairs: [z_i[0], z_j[0], z_i[1], z_j[1], ...]
    z = torch.stack((z_i, z_j), dim=1).view(2 * batch_size, -1)
    
    # Compute similarity matrix
    sim = torch.matmul(z, z.T) / tau
    
    # Compute loss for each sample
    losses = []
    for i in range(2 * batch_size):
        # Remove self-similarity
        sim_without_self = torch.cat([sim[i, :i], sim[i, i+1:]])
        log_softmax = F.log_softmax(sim_without_self, dim=0)
        # Positive pair is at adjacent index
        positive_idx = i if i % 2 == 0 else i - 1
        losses.append(log_softmax[positive_idx])
    
    return -torch.stack(losses).mean()