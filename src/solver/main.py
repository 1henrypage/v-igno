"""
Main entry point for training.
"""
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from src.solver.config import TrainingConfig
from src.solver.dgno import FoundationTrainer
from src.solver.encoder import EncoderTrainer
from src.problems import create_problem, ProblemInstance


def run_experiment(
        config: TrainingConfig,
        skip_dgno: bool = False,
        skip_nf: bool = False,
        problem: Optional[ProblemInstance] = None,
) -> Dict[str, Any]:
    """
    Run experiment from config.

    Problem instance handles its own data loading.
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
            trainer = EncoderTrainer(problem)
            trainer.setup(config, run_dir=run_dir)
            run_dir = run_dir or trainer.run_dir
            results['encoder'] = trainer.train(config)
            trainer.close()

    print(f"\nDone. Results: {run_dir}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', type=str)
    parser.add_argument('--stages', nargs='+')
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--skip-dgno', action='store_true')
    parser.add_argument('--skip-nf', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    config = TrainingConfig.load(args.config)
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