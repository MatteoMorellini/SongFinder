#!/usr/bin/env python3
"""
Unified song recognition CLI.

Usage:
    python scripts/recognize.py --approach shazam --query audio.mp3
    python scripts/recognize.py --approach shazam --query audio.mp3 --clip-length 10 --snr 5
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='SongFinder - Unified Song Recognition')
    parser.add_argument('--approach', '-a', choices=['shazam', 'grafp'], required=True,
                        help='Recognition approach to use')
    parser.add_argument('--query', '-q', type=str, required=True,
                        help='Path to query audio file')
    parser.add_argument('--db-path', type=str, default=None,
                        help='Path to database directory (default: project root)')
    parser.add_argument('--clip-length', type=float, default=None,
                        help='Clip length in seconds (for testing with shorter clips)')
    parser.add_argument('--snr', type=float, default=None,
                        help='SNR in dB for noise injection (for testing robustness)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (GraFP only)')
    
    args = parser.parse_args()
    query_path = Path(args.query)
    
    if not query_path.exists():
        print(f"Error: Query file not found: {query_path}")
        sys.exit(1)
    
    if args.approach == 'shazam':
        from approaches.shazam import ShazamRecognizer
        
        recognizer = ShazamRecognizer()
        recognizer.load()
        
        print(f"Recognizing: {query_path.name}")
        print(f"Database: {recognizer.num_indexed_songs} songs indexed")
        
        song_name, score, metadata = recognizer.recognize(
            query_path,
            clip_length_sec=args.clip_length,
            snr_db=args.snr
        )
        
        if song_name:
            print(f"\n✓ Match found: {song_name}")
            print(f"  Score: {score}")
            print(f"  Matched hashes: {metadata.get('num_matched_hashes', 'N/A')}")
        else:
            print("\n✗ No match found")
            
    elif args.approach == 'grafp':
        print("GraFP recognition requires a trained model checkpoint.")
        print("Use: python approaches/grafp/query.py --help")
        print("Or:  python approaches/grafp/generate.py to create fingerprints")
        

if __name__ == '__main__':
    main()
