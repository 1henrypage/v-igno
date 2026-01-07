"""
Base trainer with checkpoint and logging utilities.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(ABC):

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32, run_dir: Optional[Path] = None):
        self.device = device
        self.dtype = dtype
        self.run_dir = run_dir

        self.model_dict: Optional[Dict[str, nn.Module]] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.writer: Optional[SummaryWriter] = None

        self.stage_dir: Optional[Path] = None
        self.weights_dir: Optional[Path] = None
        self.tb_dir: Optional[Path] = None

    def setup_directories(self, stage_name: str) -> None:
        if self.run_dir is None:
            raise RuntimeError("run_dir must be set")
        self.stage_dir = self.run_dir / stage_name
        self.weights_dir = self.stage_dir / "weights"
        self.tb_dir = self.stage_dir / "tensorboard"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

    def setup_tensorboard(self) -> None:
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

    def log(self, tag: str, value: float, step: int) -> None:
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def save_checkpoint(self, filename: str, epoch: int, metric: Optional[float] = None,
                        metric_name: Optional[str] = None, extra: Optional[Dict] = None) -> Path:
        state = {
            'models': {name: m.state_dict() for name, m in self.model_dict.items()},
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
        }
        if metric is not None:
            state['metric'] = metric
            state['metric_name'] = metric_name
        if self.optimizer:
            state['optimizer'] = self.optimizer.state_dict()
        if self.scheduler:
            state['scheduler'] = self.scheduler.state_dict()
        if extra:
            state.update(extra)
        path = self.weights_dir / filename
        torch.save(state, path)
        return path

    @staticmethod
    def load_checkpoint(path: Path, device: Optional[torch.device] = None) -> Dict[str, Any]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location=device or 'cpu')

    def load_models_from_checkpoint(self, checkpoint: Dict[str, Any],
                                     models_to_load: Optional[list] = None, strict: bool = True) -> None:
        ckpt_models = checkpoint.get('models', checkpoint)
        to_load = models_to_load or list(self.model_dict.keys())
        for name in to_load:
            if name in self.model_dict and name in ckpt_models:
                self.model_dict[name].load_state_dict(ckpt_models[name], strict=strict)
                print(f"  Loaded: {name}")

    def train_mode(self) -> None:
        for m in self.model_dict.values():
            m.train()

    def eval_mode(self) -> None:
        for m in self.model_dict.values():
            m.eval()

    def freeze(self, model_names: list) -> None:
        for name in model_names:
            if name in self.model_dict:
                self.model_dict[name].eval()
                for p in self.model_dict[name].parameters():
                    p.requires_grad = False

    def close(self) -> None:
        if self.writer:
            self.writer.close()
            self.writer = None

    @abstractmethod
    def setup(self, config, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def train(self, **kwargs) -> None:
        raise NotImplementedError