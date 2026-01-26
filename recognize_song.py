from pathlib import Path
from modules.config import PLOT_MATCHING
from modules.matching import recognize_song
import matplotlib.pyplot as plt

if __name__ == "__main__":
    query_path = Path("test/noisier_runaway.mp3")
    song, score, matching_pairs = recognize_song(query_path)
    print("Best match:", song, "| score:", score)

    if PLOT_MATCHING and len(matching_pairs)>0:
        xs = [p[0] for p in matching_pairs]
        ys = [p[1] for p in matching_pairs]
        plt.figure(figsize=(16, 10))
        plt.scatter(xs, ys, s=1, c='blue')
        plt.xlabel("X") # time frames
        plt.ylabel("Y")
        plt.title("Scatter plot of couples")
        plt.grid(True)
        plt.show()