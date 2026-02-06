# Frequency bands (in FFT bin indices)
# n_fft=2048, sr=16kHz -> 1025 bins, limited to ~5kHz
BANDS = [
    (1, 10),      # very low
    (11, 20),     # low
    (21, 40),     # low-mid
    (41, 80),     # mid
    (81, 160),    # mid-high
    (161, 511)    # high
]

FUZ_FACTOR = 2
TARGET_SR = 16000
N_FFT = 2048
HOP_LENGTH = 368
