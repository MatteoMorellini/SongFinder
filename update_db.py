from pathlib import Path
from modules.indexing import index_folder

if __name__ == "__main__":
    #data_folder = Path("data")
    #index_folder(data_folder, DB_PATH, SONGS_DB_PATH, pattern="*.flac")
    data_folder = Path('/Users/matteomorellini/Downloads/fma_small')
    index_folder(data_folder, "fingerprints.db", "songs.db", pattern="**/*.mp3")