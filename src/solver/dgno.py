"""
Foundation Trainer: Combined DGNO (encoder-decoder) + NF training.
"""
import torch
import time
from typing import Optional, Dict, Any
from tqdm import trange
from datetime import datetime
from pathlib import Path

from src.solver.base import BaseTrainer
from src.solver.config import TrainingConfig
from src.problems import ProblemInstance
from src.utils.solver_utils import get_optimizer, get_scheduler, data_loader, var_data_loader
from src.components.nf import RealNVP


class FoundationTrainer(BaseTrainer):
    """Trains encoder-decoder (DGNO) + normalizing flow (NF)."""

    STAGE_NAME = "foundation"

    def __init__(self, problem: ProblemInstance):
        super().__init__(device=problem.device, dtype=problem.dtype)
        self.problem = problem
        self.nf = None
        self.nf_config = None  # Store for saving in checkpoint

    def _create_run_dir(self, config: TrainingConfig) -> Path:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_dir = Path(config.artifact_root) / f"{timestamp}_{config.run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def setup(self, config: TrainingConfig, pretrained_path: Optional[Path] = None) -> None:
        self.problem.pre_train_check()
        self.run_dir = self._create_run_dir(config)
        self.problem.run_dir = self.run_dir

        self.setup_directories(self.STAGE_NAME)
        self.setup_tensorboard()

        problem_models = self.problem.get_model_dict()

        # Store NF config for checkpoint
        self.nf_config = config.nf.nf
        self.nf = RealNVP(config=self.nf_config).to(self.device)

        self.model_dict = {**problem_models, 'nf': self.nf}
        for m in self.model_dict.values():
            m.to(self.device)

        ckpt_path = pretrained_path or config.get_pretrained_path()
        if ckpt_path and ckpt_path.exists():
            print(f"Loading pretrained: {ckpt_path}")
            self.load_models_from_checkpoint(self.load_checkpoint(ckpt_path, self.device), strict=False)

        config.save(self.run_dir / "config.yaml")
        print(f"Run directory: {self.run_dir}")

    def save_checkpoint(self, filename: str, epoch: int, metric: Optional[float] = None,
                        metric_name: Optional[str] = None, extra: Optional[Dict] = None) -> Path:
        """Override to include NF config."""
        state = {
            'models': {name: m.state_dict() for name, m in self.model_dict.items()},
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'nf_config': self.nf_config.to_dict() if self.nf_config else None,
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

    def train(self, config: TrainingConfig, skip_dgno: bool = False, skip_nf: bool = False) -> Dict[str, Any]:
        """Train using data from problem instance."""
        train_data = self.problem.get_train_data()
        test_data = self.problem.get_test_data()

        results = {}

        if not skip_dgno:
            print("\n" + "="*60 + "\nPHASE 1: DGNO Training\n" + "="*60)
            results['dgno'] = self._train_dgno(train_data, test_data, config)

        if not skip_nf:
            print("\n" + "="*60 + "\nPHASE 2: NF Training\n" + "="*60)
            results['nf'] = self._train_nf(train_data, test_data, config)

        print(f"\nCheckpoints saved to: {self.weights_dir}")
        return results

    def _train_dgno(self, train_data, test_data, config) -> Dict[str, Any]:
        cfg = config.dgno

        dgno_params = [p for n, m in self.model_dict.items() if n != 'nf' for p in m.parameters()]
        self.optimizer = get_optimizer(cfg.optimizer, dgno_params)
        self.scheduler = get_scheduler(cfg.scheduler, self.optimizer)

        train_loader = data_loader(train_data['a'], train_data['u'], train_data['x'], cfg.batch_size, shuffle=True)
        test_loader = data_loader(test_data['a'], test_data['u'], test_data['x'], cfg.batch_size, shuffle=False)

        t_start = time.time()
        best_error = float('inf')
        weights = cfg.loss_weights

        for epoch in trange(cfg.epochs, desc="DGNO"):
            self.train_mode()
            self.model_dict['nf'].eval()

            loss_sum, pde_sum, data_sum = 0., 0., 0.
            for a, u, x in train_loader:
                a, u, x = a.to(self.device), u.to(self.device), x.to(self.device)
                loss_pde = self.problem.loss_pde(a)
                loss_data = self.problem.loss_data(x, a, u)
                loss = loss_pde * weights.pde + loss_data * weights.data

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()
                pde_sum += loss_pde.item()
                data_sum += loss_data.item()

            self.eval_mode()
            error_sum = 0.
            with torch.no_grad():
                for a, u, x in test_loader:
                    a, u, x = a.to(self.device), u.to(self.device), x.to(self.device)
                    error_sum += self.problem.error(x, a, u).mean().item()

            n_train, n_test = len(train_loader), len(test_loader)
            avg_loss = loss_sum / n_train
            avg_error = error_sum / n_test

            self.log("dgno/loss", avg_loss, epoch)
            self.log("dgno/pde", pde_sum / n_train, epoch)
            self.log("dgno/data", data_sum / n_train, epoch)
            self.log("dgno/error", avg_error, epoch)

            if avg_error < best_error:
                best_error = avg_error
                self.save_checkpoint('best_dgno.pt', epoch, avg_error, 'error')

            if self.scheduler:
                self.scheduler.step(avg_error) if cfg.scheduler.type == 'Plateau' else self.scheduler.step()

            if (epoch + 1) % cfg.epoch_show == 0:
                print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, Error={avg_error:.4f}")

        self.save_checkpoint('last_dgno.pt', cfg.epochs - 1)
        return {'best_error': best_error, 'time': time.time() - t_start}

    def _train_nf(self, train_data, test_data, config) -> Dict[str, Any]:
        cfg = config.nf

        for n, m in self.model_dict.items():
            if n != 'nf':
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        self.optimizer = get_optimizer(cfg.optimizer, list(self.nf.parameters()))
        self.scheduler = get_scheduler(cfg.scheduler, self.optimizer)

        print("Extracting latents...")
        latents_train = self._extract_latents(train_data['a'], cfg.batch_size)
        latents_test = self._extract_latents(test_data['a'], cfg.batch_size)

        train_loader = var_data_loader(latents_train, cfg.batch_size, shuffle=True)
        test_loader = var_data_loader(latents_test, cfg.batch_size, shuffle=False)

        t_start = time.time()
        best_nll = float('inf')

        for epoch in trange(cfg.epochs, desc="NF"):
            self.nf.train()
            train_nll = 0.
            for (z,) in train_loader:
                z = z.to(self.device)
                loss = self.nf.loss(z)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_nll += loss.item()

            self.nf.eval()
            test_nll = 0.
            with torch.no_grad():
                for (z,) in test_loader:
                    test_nll += self.nf.loss(z.to(self.device)).item()

            avg_train = train_nll / len(train_loader)
            avg_test = test_nll / len(test_loader)

            self.log("nf/train_nll", avg_train, epoch)
            self.log("nf/test_nll", avg_test, epoch)

            if avg_test < best_nll:
                best_nll = avg_test
                self.save_checkpoint('best.pt', epoch, avg_test, 'nll')

            if self.scheduler:
                self.scheduler.step(avg_test) if cfg.scheduler.type == 'Plateau' else self.scheduler.step()

            if (epoch + 1) % cfg.epoch_show == 0:
                print(f"\nEpoch {epoch+1}: Train NLL={avg_train:.4f}, Test NLL={avg_test:.4f}")

        self.save_checkpoint('last.pt', cfg.epochs - 1)
        return {'best_nll': best_nll, 'time': time.time() - t_start}

    def _extract_latents(self, a: torch.Tensor, batch_size: int) -> torch.Tensor:
        enc = self.model_dict['enc']
        enc.eval()
        latents = []
        with torch.no_grad():
            for i in range(0, len(a), batch_size):
                latents.append(enc(a[i:i+batch_size].to(self.device)).cpu())
        return torch.cat(latents, dim=0)
