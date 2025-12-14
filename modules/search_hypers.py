from pathlib import Path
from modules.config import DB_PATH, SONGS_DB_PATH, PLOT_MATCHING
from modules.matching import recognize_song
import matplotlib.pyplot as plt
import random

# TODO: compute accuracy over all files, then repeat for a different config
if __name__ == "__main__":
    data_folder = Path("data")
    audio_paths = list(data_folder.glob("*.flac"))
    random.seed(42)
    audio_paths = random.sample(audio_paths, k=50)
    clip_durations = [5, 10, 15] # expressed in seconds
    SNRs_dB = [0, -1, -3]  # signal-to-noise ratios in decibels
    correct_classifications = [0] * len(clip_durations)
    for audio_path in audio_paths:
        audio_path = Path('/Users/matteomorellini/Desktop/code/shazam_clone/data/The Strokes - Is This It - 01-01 Is This It.flac')
        print(audio_path.stem)
        for i, clip_length_sec in enumerate(clip_durations):
            for snr_db in SNRs_dB:
                song, score, matching_pairs = recognize_song(audio_path, clip_length_sec=clip_length_sec, snr_db=snr_db)
                if song == audio_path.stem:
                    correct_classifications[i] += 1
                    print(f"Correctly recognized '{audio_path.stem}' with score {score} with clip length {clip_length_sec} seconds and SNR {snr_db} dB!")
                else:
                    print(f"FAILED to recognize '{audio_path.stem}'. Best match: '{song}' with score {score} with clip length {clip_length_sec} seconds and SNR {snr_db} dB.")
        break
