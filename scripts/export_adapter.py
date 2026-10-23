"""python scripts/export_adapter.py checkpoint.pt adapter.safetensors"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaloscope_style.checkpoint import export_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Style weights and architecture without optimizer/RNG state.')
    parser.add_argument('checkpoint')
    parser.add_argument('output')
    args = parser.parse_args()
    print(f'Exported step {export_checkpoint(args.checkpoint, args.output)} to {args.output}')
