"""
Peak extractor for GraFP - extracts spectral peak features with positional encoding.
"""

import torch
import torch.nn as nn


class GPUPeakExtractorv2(nn.Module):
    """Convolutional peak extractor with time-frequency positional encoding."""
    
    def __init__(self, cfg):
        super().__init__()
        self.n_filters = cfg['n_filters']
        self.stride = cfg['peak_stride']
        kernel = cfg['blur_kernel']
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, self.n_filters, kernel_size=kernel,
                      stride=(self.stride, 1),
                      padding=(kernel[0] // 2, kernel[1] // 2)),
            nn.ReLU()
        )
        
        n_gpus = max(1, torch.cuda.device_count())
        batch_per_gpu = cfg['bsz_train'] // n_gpus
        
        # Positional encodings (time and frequency)
        self.register_buffer('T_pos', self._make_time_pos(cfg['n_frames'], cfg['n_mels'], batch_per_gpu))
        self.register_buffer('F_pos', self._make_freq_pos(cfg['n_frames'], cfg['n_mels'], batch_per_gpu))
        
        self._init_weights()
    
    def _make_time_pos(self, n_frames, n_mels, batch_size):
        t = torch.linspace(0, 1, steps=n_frames)
        return t.view(1, 1, n_frames).expand(batch_size, n_mels, -1)
    
    def _make_freq_pos(self, n_frames, n_mels, batch_size):
        f = torch.linspace(0, 1, steps=n_mels)
        return f.view(1, n_mels, 1).expand(batch_size, -1, n_frames)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, spec):
        # Normalize spectrogram
        min_v = spec.amin(dim=(1, 2), keepdim=True)
        max_v = spec.amax(dim=(1, 2), keepdim=True)
        spec = (spec - min_v) / (max_v - min_v + 1e-8)
        
        # Add channel dimension
        spec = spec.unsqueeze(1)
        
        # Get positional encodings for this batch size
        B, _, H, W = spec.shape
        if B != self.T_pos.shape[0] or H != self.T_pos.shape[1] or W != self.T_pos.shape[2]:
            T_pos = torch.linspace(0, 1, steps=W, device=spec.device).view(1, 1, W).expand(B, H, -1)
            F_pos = torch.linspace(0, 1, steps=H, device=spec.device).view(1, H, 1).expand(B, -1, W)
        else:
            T_pos = self.T_pos[:B]
            F_pos = self.F_pos[:B]
        
        # Concatenate: [time_pos, freq_pos, spectrogram]
        x = torch.cat([T_pos.unsqueeze(1), F_pos.unsqueeze(1), spec], dim=1)
        
        # Apply convolution
        x = self.conv(x)
        
        # Reshape to (B, C, N) where N = H*W
        return x.reshape(x.shape[0], x.shape[1], -1)