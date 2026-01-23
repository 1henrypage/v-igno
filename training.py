#!/usr/bin/env python3
"""
Main entry point for IGNO training.

Trains encoder + decoders + NF jointly in a single phase.
Models are owned by ProblemInstance. Trainer orchestrates training.

Usage:
    python train.py --config configs/training/example_train.yaml
    python train.py --config configs/training/example_train.yaml --device cuda:1
    python train.py --config configs/training/example_train.yaml --epochs 5000
    python train.py --config configs/training/example_train.yaml --pretrained path/to/checkpoint.pt
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from src.solver.config import TrainingConfig
from src.solver.trainer import IGNOTrainer
from src.problems import create_problem, ProblemInstance


def run_training(
        config: TrainingConfig,
        problem: Optional[ProblemInstance] = None,
        pretrained_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run IGNO training.

    Args:
        config: Training configuration
        problem: Optional pre-created problem instance
        pretrained_path: Optional path to pretrained checkpoint

    Returns:
        Results dict with metrics
    """
    if problem is None:
        print(f"Creating problem: {config.problem.type}")
        problem = create_problem(config)

    trainer = IGNOTrainer(problem)
    trainer.setup(config, pretrained_path=pretrained_path)

    results = trainer.train(config)
    trainer.close()

    print(f"\nDone. Results saved to: {trainer.run_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Train IGNO models')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--device', type=str, help='Override device')
    parser.add_argument('--pretrained', type=str, help='Path to pretrained checkpoint')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--standardize-latent', action='store_true',
                        help='Enable latent standardization for NF')
    parser.add_argument('--no-standardize-latent', action='store_true',
                        help='Disable latent standardization for NF')
    parser.add_argument('--dry-run', action='store_true', help='Print config and exit')

    args = parser.parse_args()

    # Load config
    config = TrainingConfig.load(args.config)

    # Apply CLI overrides
    if args.device:
        config.device = args.device
    if args.pretrained:
        config.pretrained = {'path': args.pretrained}
    if args.seed:
        config.seed = args.seed
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.optimizer.lr = args.lr
    if args.standardize_latent:
        config.training.standardize_latent = True
    if args.no_standardize_latent:
        config.training.standardize_latent = False

    print(f"Config: {args.config}")
    print(f"Device: {config.device}, Seed: {config.seed}")
    print(f"Problem: {config.problem.type}")
    print(f"Epochs: {config.training.epochs}, Batch size: {config.training.batch_size}")
    print(f"Standardize latent: {config.training.standardize_latent}")
    print(f"Loss weights: pde={config.training.loss_weights.pde}, data={config.training.loss_weights.data}")

    if args.dry_run:
        print("\n[DRY RUN] Config loaded successfully. Exiting.")
        return

    # Get pretrained path if specified
    pretrained_path = None
    if args.pretrained:
        pretrained_path = Path(args.pretrained)

    run_training(config, pretrained_path=pretrained_path)


if __name__ == '__main__':
    main()
