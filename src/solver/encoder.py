"""
Encoder Trainer: Stage 2 - trains new encoder using pretrained foundation.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from src.solver.base import BaseTrainer
from src.solver.config import TrainingConfig
from src.problems import ProblemInstance
from src.components.nf import RealNVP


class EncoderTrainer(BaseTrainer):
    """Stage 2: Train new encoder with frozen decoder + NF."""

    STAGE_NAME = "encoder"

    def __init__(self, problem: ProblemInstance):
        super().__init__(device=problem.device, dtype=problem.dtype)
        self.problem = problem
        self.new_encoder = None

    def setup(self, config: TrainingConfig, pretrained_path: Optional[Path] = None,
              run_dir: Optional[Path] = None) -> None:
        ckpt_path = pretrained_path or config.get_pretrained_path()
        if ckpt_path is None or not ckpt_path.exists():
            raise ValueError("EncoderTrainer requires pretrained foundation weights")

        if run_dir:
            self.run_dir = Path(run_dir)
        else:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.run_dir = Path(config.artifact_root) / f"{timestamp}_{config.run_name}_encoder"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.setup_directories(self.STAGE_NAME)
        self.setup_tensorboard()

        print(f"Loading pretrained: {ckpt_path}")
        checkpoint = self.load_checkpoint(ckpt_path, self.device)
        ckpt_models = checkpoint.get('models', checkpoint)

        problem_models = self.problem.get_model_dict()
        decoder = problem_models['dec']
        decoder.load_state_dict(ckpt_models['dec'])

        nf = RealNVP(config=config.nf.nf)
        nf.load_state_dict(ckpt_models['nf'])

        self.new_encoder = self._create_encoder(config)

        self.model_dict = {'new_enc': self.new_encoder, 'dec': decoder.to(self.device), 'nf': nf.to(self.device)}

        if config.encoder.freeze_decoder:
            self.freeze(['dec'])
        if config.encoder.freeze_nf:
            self.freeze(['nf'])

        config.save(self.run_dir / "config.yaml")

    def _create_encoder(self, config: TrainingConfig) -> nn.Module:
        raise NotImplementedError("Implement _create_encoder()")

    def train(self, config: TrainingConfig) -> Dict[str, Any]:
        """Train using data from problem instance."""
        train_data = self.problem.get_train_data()
        test_data = self.problem.get_test_data()
        raise NotImplementedError("Implement train()")