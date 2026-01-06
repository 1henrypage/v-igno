#!/usr/bin/env python3
"""
HPC training script.

Usage:
    python run_training.py --config configs/experiment.yaml
    python run_training.py --config configs/experiment.yaml --device cuda:0
    python run_training.py --config configs/experiment.yaml --seed 42
    python run_training.py --config configs/experiment.yaml --skip-nf
    python run_training.py --config configs/experiment.yaml --pretrained /path/to/best.pt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.solver.main import main

if __name__ == '__main__':
    main()
