"""
Training and evaluation configuration. Single YAML file.

NF architecture is hardcoded per-problem (in _build_models), not in config.
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


# =============================================================================
# Problem Config
# =============================================================================

@dataclass
class ProblemConfig(BaseConfig):
    """Problem type and data paths."""
    type: str
    train_data: Optional[str] = None
    test_data: Optional[str] = None


# =============================================================================
# Training Components
# =============================================================================

@dataclass
class OptimizerConfig(BaseConfig):
    type: Literal['Adam', 'AdamW', 'RMSprop', 'SGD'] = None
    lr: float = None
    weight_decay: float = None


@dataclass
class SchedulerConfig(BaseConfig):
    type: Optional[Literal['StepLR', 'Plateau', 'CosineAnnealing']] = None
    step_size: Optional[int] = None
    gamma: Optional[float] = None
    patience: Optional[int] = None
    factor: Optional[float] = None
    total_steps: Optional[int] = None
    pct_start: Optional[float] = None
    anneal_strategy: Optional[str] = None
    div_factor: Optional[int] = None
    final_div_factor: Optional[int] = None


@dataclass
class LossWeights(BaseConfig):
    pde: float = None
    data: float = None


# =============================================================================
# Training Phase Configs
# =============================================================================

@dataclass
class DGNOConfig(BaseConfig):
    """Config for DGNO (encoder-decoder) training phase."""
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
    """Config for NF training phase (architecture is in problem, not here)."""
    epochs: int = 1000
    batch_size: int = 100
    epoch_show: int = 100
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
            optimizer=OptimizerConfig.from_dict(data.get('optimizer', {})),
            scheduler=SchedulerConfig.from_dict(data.get('scheduler', {})),
        )


@dataclass
class EncoderConfig(BaseConfig):
    """Config for encoder training phase (if used separately)."""
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


# =============================================================================
# Evaluation Configs
# =============================================================================

@dataclass
class InversionConfig(BaseConfig):
    """Config for gradient-based inversion."""
    epochs: int = 1000
    loss_weights: LossWeights =  None
    optimizer: OptimizerConfig = None
    scheduler: SchedulerConfig = None

    @classmethod
    def from_dict(cls, data: dict) -> 'InversionConfig':
        if data is None:
            return cls()
        return cls(
            epochs=data.get('epochs', 1000),
            loss_weights=LossWeights.from_dict(data.get('loss_weights', {})),
            optimizer=OptimizerConfig.from_dict(data.get('optimizer', {})),
            scheduler=SchedulerConfig.from_dict(data.get('scheduler', {})),
        )


@dataclass
class EvaluationConfig(BaseConfig):
    """Config for evaluation/inversion."""
    method: Literal['igno', 'encoder'] = 'igno'

    # Batch size
    batch_size: int = None

    # Observation setup
    n_obs: int = 100
    obs_sampling: Literal['random', 'grid'] = 'random'

    # Noise (None for clean)
    snr_db: Optional[float] = None

    # Inversion params
    inversion: InversionConfig = field(default_factory=InversionConfig)

    # Output
    results_dir: str = "results"

    @classmethod
    def from_dict(cls, data: dict) -> 'EvaluationConfig':
        if data is None:
            return cls()
        return cls(
            method=data.get('method', 'igno'),
            batch_size=data.get('batch_size', None),
            n_obs=data.get('n_obs', 100),
            obs_sampling=data.get('obs_sampling', 'random'),
            snr_db=data.get('snr_db', 25.0),
            inversion=InversionConfig.from_dict(data.get('inversion', {})),
            results_dir=data.get('results_dir', 'results'),
        )


# =============================================================================
# Main Config
# =============================================================================

@dataclass
class TrainingConfig(BaseConfig):
    """Main config for training and evaluation."""
    run_name: str
    device: str
    artifact_root: str = "runs"
    seed: int = 10086

    problem: ProblemConfig = field(default_factory=ProblemConfig)
    stages: List[str] = field(default_factory=lambda: ["foundation"])
    pretrained: Optional[Dict[str, Any]] = None

    # Training phase configs
    dgno: DGNOConfig = field(default_factory=DGNOConfig)
    nf: NFTrainConfig = field(default_factory=NFTrainConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    # Evaluation config
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: dict) -> 'TrainingConfig':
        if data is None:
            return cls()

        problem_data = data.get('problem', {})
        if isinstance(problem_data, str):
            problem_data = {'type': problem_data}

        return cls(
            run_name=data.get('run_name'),
            device=data.get('device'),
            artifact_root=data.get('artifact_root', 'runs'),
            seed=data.get('seed', 10086),
            problem=ProblemConfig.from_dict(problem_data),
            stages=data.get('stages', ['foundation']),
            pretrained=data.get('pretrained'),
            dgno=DGNOConfig.from_dict(data.get('dgno', {})),
            nf=NFTrainConfig.from_dict(data.get('nf', {})),
            encoder=EncoderConfig.from_dict(data.get('encoder', {})),
            evaluation=EvaluationConfig.from_dict(data.get('evaluation', {})),
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
            raise RuntimeError("Pretrained path doesn't exist.")
        if path.suffix == '.pt':
            return path
        stage = self.pretrained.get('stage', 'foundation')
        checkpoint = self.pretrained.get('checkpoint', 'best.pt')
        return path / stage / 'weights' / checkpoint
