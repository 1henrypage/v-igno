
"""Solver module."""
from .config import TrainingConfig, DGNOConfig, NFTrainConfig, EncoderConfig
from .dgno import FoundationTrainer
from .encoder import EncoderTrainer
from src.problems import ProblemInstance, create_problem, register_problem, list_problems

__all__ = [
    'TrainingConfig', 'DGNOConfig', 'NFTrainConfig', 'EncoderConfig',
    'FoundationTrainer', 'EncoderTrainer',
    'ProblemInstance', 'create_problem', 'register_problem', 'list_problems',
]
