import os
import pickle

def load_db(path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

def save_db(path: str, table):
    with open(path, "wb") as f:
        pickle.dump(table, f)

def get_song_id(song_table: dict, name: str) -> int:
    """Get or create song ID. Returns existing ID if song already indexed."""
    if name in song_table:
        return song_table[name]
    new_id = len(song_table) + 1
    song_table[name] = new_id
    return new_id
