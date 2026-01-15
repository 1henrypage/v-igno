"""
Main entry point for training.

Models are owned by ProblemInstance. Trainers orchestrate training phases.
"""
import argparse
from typing import Optional, Dict, Any

from src.solver.config import TrainingConfig
from src.solver.dgno import FoundationTrainer
from src.problems import create_problem, ProblemInstance

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def run_experiment(
        config: TrainingConfig,
        skip_dgno: bool = False,
        skip_nf: bool = False,
        problem: Optional[ProblemInstance] = None,
) -> Dict[str, Any]:
    """
    Run experiment from config.

    Problem instance handles its own data loading and model building.

    Args:
        config: Training configuration
        skip_dgno: Skip DGNO training phase
        skip_nf: Skip NF training phase
        problem: Optional pre-created problem instance

    Returns:
        Results dict with metrics from each stage
    """
    if problem is None:
        print(f"Creating problem: {config.problem.type}")
        problem = create_problem(config)

    results = {}
    run_dir = None

    for stage in config.stages:
        print(f"\n{'=' * 60}\nSTAGE: {stage.upper()}\n{'=' * 60}")

        if stage == 'foundation':
            trainer = FoundationTrainer(problem)
            trainer.setup(config)
            run_dir = trainer.run_dir
            results['foundation'] = trainer.train(config, skip_dgno, skip_nf)
            trainer.close()

        elif stage == 'encoder':
            # EncoderTrainer would follow similar pattern
            # For now, raise not implemented
            raise NotImplementedError("Encoder stage not yet implemented in new design")

    print(f"\nDone. Results: {run_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Train IGNO models')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--device', type=str, help='Override device')
    parser.add_argument('--stages', nargs='+', help='Stages to run')
    parser.add_argument('--pretrained', type=str, help='Path to pretrained checkpoint')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--skip-dgno', action='store_true', help='Skip DGNO training')
    parser.add_argument('--skip-nf', action='store_true', help='Skip NF training')
    parser.add_argument('--dry-run', action='store_true', help='Print config and exit')

    args = parser.parse_args()

    # Load config
    config = TrainingConfig.load(args.config)

    # Apply CLI overrides
    if args.device:
        config.device = args.device
    if args.stages:
        config.stages = args.stages
    if args.pretrained:
        config.pretrained = {'path': args.pretrained}
    if args.seed:
        config.seed = args.seed

    print(f"Config: {args.config}")
    print(f"Device: {config.device}, Seed: {config.seed}")
    print(f"Stages: {config.stages}, Problem: {config.problem.type}")

    if args.dry_run:
        print("[DRY RUN] Done.")
        return

    run_experiment(config, args.skip_dgno, args.skip_nf)


if __name__ == '__main__':
    main()

