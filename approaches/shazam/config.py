# ---------- CONFIG ---------- #

DB_PATH = "fingerprints.db"
SONGS_DB_PATH = "songs.db"
PLOT_SPECTROGRAM = False
PLOT_MATCHING = True

# Define frequency bands (in terms of frequency bin indices)
# n_fft = 2048 -> freq bins = 1025 (0 to 1024) but we will limit to ~5kHz
BANDS = [
    (1, 10),      # very low
    (11, 20),     # low
    (21, 40),     # low-mid
    (41, 80),     # mid
    (81, 160),    # mid-high
    (161, 511)    # high
]

FUZ_FACTOR = 2  # absorb small variations in frequency / time: 43 → 42, 21 → 20, etc.
TARGET_SR = 11025
N_FFT = 2048
HOP_LENGTH = 368