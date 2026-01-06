
"""Solver module."""
from .config import TrainingConfig, DGNOConfig, NFTrainConfig, EncoderConfig, NFConfig
from .base import BaseTrainer
from .dgno import FoundationTrainer, Solver
from .encoder import EncoderTrainer
from .main import run_experiment
from src.problems import ProblemInstance, create_problem, register_problem, list_problems

__all__ = [
    'TrainingConfig', 'DGNOConfig', 'NFTrainConfig', 'EncoderConfig', 'NFConfig',
    'BaseTrainer', 'FoundationTrainer', 'Solver', 'EncoderTrainer',
    'run_experiment', 'ProblemInstance', 'create_problem', 'register_problem', 'list_problems',
]