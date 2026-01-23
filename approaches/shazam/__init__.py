"""
Shazam-style song recognition approach.

This approach follows the classic Shazam algorithm:
1. Extract spectrogram from audio
2. Find spectral peaks in frequency bands
3. Create constellation hashes from peak pairs
4. Match against a database of fingerprints
"""

from .recognizer import ShazamRecognizer
from .config import BANDS, TARGET_SR, N_FFT, HOP_LENGTH, FUZ_FACTOR

__all__ = ['ShazamRecognizer', 'BANDS', 'TARGET_SR', 'N_FFT', 'HOP_LENGTH', 'FUZ_FACTOR']
