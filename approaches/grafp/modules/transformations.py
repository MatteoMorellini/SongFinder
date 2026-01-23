"""
Audio transformations for GraFP inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


class AudioTransform(nn.Module):
    """Audio to mel-spectrogram transformation for inference."""
    
    def __init__(self, cfg):
        super().__init__()
        self.sample_rate = cfg['fs']
        self.overlap = cfg['overlap']
        self.n_frames = cfg['n_frames']
        
        self.logmelspec = nn.Sequential(
            MelSpectrogram(
                sample_rate=self.sample_rate,
                win_length=cfg['win_len'],
                hop_length=cfg['hop_len'],
                n_fft=cfg['n_fft'],
                n_mels=cfg['n_mels']
            ),
            AmplitudeToDB()
        )
    
    def forward(self, audio):
        """Transform audio to overlapping mel-spectrogram segments."""
        spec = self.logmelspec(audio.squeeze(0)).transpose(1, 0)
        segments = spec.unfold(0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap)))
        return segments