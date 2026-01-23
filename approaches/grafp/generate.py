"""
GraFP fingerprint generation from pre-trained model.
"""

import argparse
import torch

from approaches.grafp.util import load_config
from approaches.grafp.modules.data import NeuralfpDataset
from approaches.grafp.modules.transformations import AudioTransform
from approaches.grafp.inference import load_model, extract_fingerprints

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description='Generate fingerprints with pre-trained GraFP')
    parser.add_argument('--config', default='config/grafp.yaml', type=str)
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--audio_dir', required=True, type=str, help='Path to audio files')
    parser.add_argument('--output_dir', default='output', type=str)
    parser.add_argument('--k', default=3, type=int, help='K for graph encoder')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    model = load_model(cfg, args.checkpoint, args.k)
    transform = AudioTransform(cfg).to(DEVICE)
    
    dataset = NeuralfpDataset(cfg, args.audio_dir, train=False, inference=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    n_items = extract_fingerprints(loader, model, transform, args.output_dir)
    print(f"Generated {n_items} fingerprints in {args.output_dir}")


if __name__ == '__main__':
    main()