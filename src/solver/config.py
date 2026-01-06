"""
Training configuration. Single YAML file.
"""
from dataclasses import dataclass, asdict, field, fields
from typing import Literal, Optional, List, Dict, Any, Type, TypeVar
from pathlib import Path
import yaml
import torch
import numpy as np


def _serialize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, torch.Tensor)):
        return float(obj)
    elif isinstance(obj, Path):
        return str(obj)
    return obj


T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        if data is None:
            data = {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(asdict(self))


@dataclass
class ProblemConfig(BaseConfig):
    type: str = "darcy_flow_continuous"


@dataclass
class OptimizerConfig(BaseConfig):
    type: Literal['Adam', 'AdamW', 'RMSprop', 'SGD'] = 'Adam'
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig(BaseConfig):
    type: Optional[Literal['StepLR', 'Plateau', 'CosineAnnealing']] = None
    step_size: int = 100
    gamma: float = 0.5
    patience: int = 10
    factor: float = 0.5


@dataclass
class LossWeights(BaseConfig):
    pde: float = 1.0
    data: float = 1.0


@dataclass
class NFConfig(BaseConfig):
    dim: int = 64
    num_flows: int = 8
    hidden_dim: int = 128
    num_layers: int = 2


@dataclass
class DGNOConfig(BaseConfig):
    epochs: int = 1000
    batch_size: int = 100
    epoch_show: int = 100
    loss_weights: LossWeights = field(default_factory=LossWeights)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @classmethod
    def from_dict(cls, data: dict) -> 'DGNOConfig':
        if data is None:
            return cls()
        return cls(
            epochs=data.get('epochs', 1000),
            batch_size=data.get('batch_size', 100),
            epoch_show=data.get('epoch_show', 100),
            loss_weights=LossWeights.from_dict(data.get('loss_weights', {})),
            optimizer=OptimizerConfig.from_dict(data.get('optimizer', {})),
            scheduler=SchedulerConfig.from_dict(data.get('scheduler', {})),
        )


@dataclass
class NFTrainConfig(BaseConfig):
    epochs: int = 1000
    batch_size: int = 100
    epoch_show: int = 100
    nf: NFConfig = field(default_factory=NFConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @classmethod
    def from_dict(cls, data: dict) -> 'NFTrainConfig':
        if data is None:
            return cls()
        return cls(
            epochs=data.get('epochs', 1000),
            batch_size=data.get('batch_size', 100),
            epoch_show=data.get('epoch_show', 100),
            nf=NFConfig.from_dict(data.get('nf', {})),
            optimizer=OptimizerConfig.from_dict(data.get('optimizer', {})),
            scheduler=SchedulerConfig.from_dict(data.get('scheduler', {})),
        )


@dataclass
class EncoderConfig(BaseConfig):
    epochs: int = 1000
    batch_size: int = 100
    epoch_show: int = 100
    freeze_decoder: bool = True
    freeze_nf: bool = True
    loss_weights: LossWeights = field(default_factory=LossWeights)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @classmethod
    def from_dict(cls, data: dict) -> 'EncoderConfig':
        if data is None:
            return cls()
        return cls(
            epochs=data.get('epochs', 1000),
            batch_size=data.get('batch_size', 100),
            epoch_show=data.get('epoch_show', 100),
            freeze_decoder=data.get('freeze_decoder', True),
            freeze_nf=data.get('freeze_nf', True),
            loss_weights=LossWeights.from_dict(data.get('loss_weights', {})),
            optimizer=OptimizerConfig.from_dict(data.get('optimizer', {})),
            scheduler=SchedulerConfig.from_dict(data.get('scheduler', {})),
        )


@dataclass
class TrainingConfig(BaseConfig):
    """Main config."""
    run_name: str = "experiment"
    device: str = "cuda"
    artifact_root: str = "runs"
    seed: int = 10086

    problem: ProblemConfig = field(default_factory=ProblemConfig)
    stages: List[str] = field(default_factory=lambda: ["foundation"])
    pretrained: Optional[Dict[str, Any]] = None

    dgno: DGNOConfig = field(default_factory=DGNOConfig)
    nf: NFTrainConfig = field(default_factory=NFTrainConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    @classmethod
    def from_dict(cls, data: dict) -> 'TrainingConfig':
        if data is None:
            return cls()

        problem_data = data.get('problem', {})
        if isinstance(problem_data, str):
            problem_data = {'type': problem_data}

        return cls(
            run_name=data.get('run_name', 'experiment'),
            device=data.get('device', 'cuda'),
            artifact_root=data.get('artifact_root', 'runs'),
            seed=data.get('seed', 10086),
            problem=ProblemConfig.from_dict(problem_data),
            stages=data.get('stages', ['foundation']),
            pretrained=data.get('pretrained'),
            dgno=DGNOConfig.from_dict(data.get('dgno', {})),
            nf=NFTrainConfig.from_dict(data.get('nf', {})),
            encoder=EncoderConfig.from_dict(data.get('encoder', {})),
        )

    @classmethod
    def load(cls, path: Path) -> 'TrainingConfig':
        with open(Path(path), 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def get_pretrained_path(self) -> Optional[Path]:
        if self.pretrained is None:
            return None
        path = Path(self.pretrained.get('path', ''))
        if not path.exists():
            return None
        if path.suffix == '.pt':
            return path
        stage = self.pretrained.get('stage', 'foundation')
        checkpoint = self.pretrained.get('checkpoint', 'best.pt')
        return path / stage / 'weights' / checkpoint