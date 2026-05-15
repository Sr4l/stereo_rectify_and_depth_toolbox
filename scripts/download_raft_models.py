#!/usr/bin/env python3
"""Download pretrained RAFT-Stereo models."""

import os
import argparse
import sys

try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False

MODELS = {
    'middlebury': {
        'url': 'https://drive.google.com/file/d/1m3KoukUmKDoMv-ySOO6vBzYfWLyj9yqd/view',
        'file': 'raftstereo-middlebury.pth',
        'desc': 'Best for in-the-wild images (RECOMMENDED)'
    },
    'eth3d': {
        'url': 'https://drive.google.com/file/d/1-27_cej_dqb3B5ONen25d8ZGNrF2GFHH/view',
        'file': 'raftstereo-eth3d.pth',
        'desc': 'ETH3D dataset - high resolution stereo'
    },
    'sceneflow': {
        'url': 'https://drive.google.com/file/d/1WqCGS2DikrBE8Ax0Z52pTtyCLvkt8lR5/view',
        'file': 'raftstereo-sceneflow.pth',
        'desc': 'SceneFlow (FlyingThings3D, Driving, Monkaa)'
    },
    'realtime': {
        'url': 'https://drive.google.com/file/d/1iIMunuys09OE4pfSiiP4HXa4lW33SeMT/view',
        'file': 'raftstereo-realtime.pth',
        'desc': 'Fastest model for real-time applications'
    }
}

def download_model(model_name, output_dir='models'):
    if not GDOWN_AVAILABLE:
        print("Error: gdown not installed. Install with: pip install gdown")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    if model_name not in MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False
    
    model = MODELS[model_name]
    output_path = os.path.join(output_dir, model['file'])
    
    print(f"\n{'='*60}")
    print(f"Downloading RAFT-Stereo: {model_name}")
    print(f"Description: {model['desc']}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    try:
        file_id = model['url'].split('/d/')[1].split('/view')[0]
        url = f'https://drive.google.com/uc?id={file_id}'
        
        gdown.download(url, output_path, quiet=False)
        
        print(f"\n✓ Successfully downloaded: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nYou can also download models manually from:")
        print("https://drive.google.com/drive/folders/1booUFYEXmsdombVuglatP0nZXb5qI89J")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download pretrained RAFT-Stereo models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_raft_models.py --model middlebury
  python download_raft_models.py -m eth3d --output ./my_models

Manual Download:
  If automatic download fails, you can manually download from:
  https://drive.google.com/drive/folders/1booUFYEXmsdombVuglatP0nZXb5qI89J
  
  Then place the .pth file in the models/ directory.
        """
    )
    parser.add_argument(
        '--model', '-m',
        default='middlebury',
        choices=['middlebury', 'eth3d', 'sceneflow', 'realtime'],
        help='Model to download (default: middlebury)'
    )
    parser.add_argument(
        '--output', '-o',
        default='models',
        help='Output directory (default: models)'
    )
    
    args = parser.parse_args()
    
    success = download_model(args.model, args.output)
    sys.exit(0 if success else 1)
