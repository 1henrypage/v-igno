"""Solver module."""
from .config import TrainingConfig, IGNOTrainingConfig, EvaluationConfig, InversionConfig
from .trainer import IGNOTrainer
from src.problems import ProblemInstance, create_problem, register_problem, list_problems

__all__ = [
    'TrainingConfig', 'IGNOTrainingConfig', 'EvaluationConfig', 'InversionConfig',
    'IGNOTrainer',
    'ProblemInstance', 'create_problem', 'register_problem', 'list_problems',
]
