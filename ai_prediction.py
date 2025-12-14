from pathlib import Path
from modules.config import DB_PATH, SONGS_DB_PATH
from modules.graph import train_graph

if __name__ == "__main__":
    data_folder = Path("data")
    train_graph(data_folder, pattern="*.flac")