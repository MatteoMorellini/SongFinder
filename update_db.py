from pathlib import Path
from modules.config import DB_PATH, SONGS_DB_PATH
from modules.indexing import index_folder

if __name__ == "__main__":
    data_folder = Path("data")
    index_folder(data_folder, DB_PATH, SONGS_DB_PATH, pattern="*.flac")
