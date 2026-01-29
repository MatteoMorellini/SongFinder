"""
SimCLR model wrapper for GraFP - contrastive learning framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from approaches.grafp.peak_extractor import GPUPeakExtractorv2


class SimCLR(nn.Module):
    """SimCLR contrastive learning framework with GraFP encoder."""
    
    def __init__(self, cfg, encoder):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        
        d = cfg['d']
        h = cfg['h']
        u = cfg['u']
        
        self.peak_extractor = GPUPeakExtractorv2(cfg) if cfg['arch'] == 'grafp' else None
        
        self.projector = nn.Sequential(
            nn.Linear(h, d * u),
            nn.ELU(),
            nn.Linear(d * u, d)
        )
    
    def forward(self, x_i, x_j):
        if self.peak_extractor:
            x_i = self.peak_extractor(x_i)
            x_j = self.peak_extractor(x_j)
        
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        
        z_i = F.normalize(self.projector(h_i), p=2)
        z_j = F.normalize(self.projector(h_j), p=2)
        
        return h_i, h_j, z_i, z_j