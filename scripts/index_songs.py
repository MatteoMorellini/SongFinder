#!/usr/bin/env python3
"""
Index songs into the database.

Usage:
    python scripts/index_songs.py --approach shazam --folder data/ --pattern "*.flac"
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='SongFinder - Index Songs')
    parser.add_argument('--approach', '-a', choices=['shazam', 'grafp'], required=True,
                        help='Recognition approach to use')
    parser.add_argument('--folder', '-f', type=str, required=True,
                        help='Path to folder containing audio files')
    parser.add_argument('--pattern', '-p', type=str, default='*.flac',
                        help='Glob pattern for audio files (default: *.flac)')
    parser.add_argument('--db-path', type=str, default=None,
                        help='Path to save database (default: project root)')
    
    args = parser.parse_args()
    folder = Path(args.folder)
    
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)
    
    if args.approach == 'shazam':
        from approaches.shazam import ShazamRecognizer
        
        recognizer = ShazamRecognizer()
        
        # Try to load existing database
        try:
            recognizer.load()
            print(f"Loaded existing database: {recognizer.num_indexed_songs} songs")
        except:
            print("Starting with empty database")
        
        print(f"\nIndexing songs from: {folder}")
        print(f"Pattern: {args.pattern}")
        
        count = recognizer.index_folder(folder, pattern=args.pattern)
        recognizer.save()
        
        print(f"\n✓ Indexed {count} new songs")
        print(f"  Total songs in database: {recognizer.num_indexed_songs}")
        
    elif args.approach == 'grafp':
        print("GraFP indexing requires training a model first.")
        print("Use: python approaches/grafp/train.py --help")
        print("Then: python approaches/grafp/generate.py to create fingerprints")


if __name__ == '__main__':
    main()
