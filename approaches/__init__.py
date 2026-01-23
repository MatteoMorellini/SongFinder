"""
SongFinder - Song Recognition Approaches

This package provides two approaches for audio fingerprinting and song recognition:
1. Shazam-style: Traditional signal processing with spectral peaks and constellation hashing
2. GraFP: Graph Neural Network based approach using contrastive learning
"""

from approaches.base import BaseSongRecognizer

__all__ = ['BaseSongRecognizer']
