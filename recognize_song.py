from pathlib import Path
from modules.config import DB_PATH, SONGS_DB_PATH
from modules.matching import recognize_song

if __name__ == "__main__":
    query_path = Path("test/noisier_runaway.mp3")
    song, score = recognize_song(query_path, DB_PATH, SONGS_DB_PATH)
    print("Best match:", song, "| score:", score)
