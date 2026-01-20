"""
Foundation Trainer: Combined DGNO (encoder-decoder) + NF training.

This trainer orchestrates training but does NOT own models.
All models are owned by the ProblemInstance.
"""
import torch
import time
from typing import Optional, Dict, Any
from tqdm import trange
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from src.solver.config import TrainingConfig
from src.problems import ProblemInstance
from src.utils.solver_utils import get_optimizer, get_scheduler, data_loader, var_data_loader


class FoundationTrainer:
    """
    Trains encoder-decoder (DGNO) + normalizing flow (NF).

    Models are owned by the ProblemInstance - this class just orchestrates training.
    """

    STAGE_NAME = "foundation"

    def __init__(self, problem: ProblemInstance):
        self.problem = problem
        self.device = problem.device
        self.dtype = problem.dtype

        # Training state (ephemeral)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.writer: Optional[SummaryWriter] = None

        # Directories
        self.run_dir: Optional[Path] = None
        self.stage_dir: Optional[Path] = None
        self.weights_dir: Optional[Path] = None
        self.tb_dir: Optional[Path] = None

    def _create_run_dir(self, config: TrainingConfig) -> Path:
        """Create timestamped run directory."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_dir = Path(config.artifact_root) / f"{timestamp}_{config.run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _setup_directories(self) -> None:
        """Setup stage directories."""
        if self.run_dir is None:
            raise RuntimeError("run_dir must be set before calling _setup_directories")

        self.stage_dir = self.run_dir / self.STAGE_NAME
        self.weights_dir = self.stage_dir / "weights"
        self.tb_dir = self.stage_dir / "tensorboard"

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

    def _setup_tensorboard(self) -> None:
        """Initialize TensorBoard writer."""
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

    def _log(self, tag: str, value: float, step: int) -> None:
        """Log to TensorBoard."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def setup(self, config: TrainingConfig, pretrained_path: Optional[Path] = None) -> None:
        """
        Setup trainer: create directories, load pretrained if specified.

        Args:
            config: Training configuration
            pretrained_path: Optional path to pretrained checkpoint
        """
        self.problem.pre_train_check()

        # Create run directory
        self.run_dir = self._create_run_dir(config)
        self.problem.run_dir = self.run_dir

        # Setup directories and tensorboard
        self._setup_directories()
        self._setup_tensorboard()

        # Load pretrained if specified
        ckpt_path = pretrained_path or config.get_pretrained_path()
        if ckpt_path:
            if not ckpt_path.exists():
                raise RuntimeError("Couldn't find pretrained checkpoint")
            print(f"Loading pretrained: {ckpt_path}")
            self.problem.load_checkpoint(ckpt_path, strict=False)


        # Save config
        config.save(self.run_dir / "config.yaml")
        print(f"Run directory: {self.run_dir}")

    def train(self, config: TrainingConfig, skip_dgno: bool = False, skip_nf: bool = False) -> Dict[str, Any]:
        """
        Train DGNO and NF phases.

        Args:
            config: Training configuration
            skip_dgno: Skip DGNO training phase
            skip_nf: Skip NF training phase

        Returns:
            Results dict with metrics from each phase
        """
        train_data = self.problem.get_train_data()
        test_data = self.problem.get_test_data()

        results = {}

        if not skip_dgno:
            print("\n" + "=" * 60 + "\nPHASE 1: DGNO Training\n" + "=" * 60)
            results['dgno'] = self._train_dgno(train_data, test_data, config)

        if not skip_nf:
            # IMPORTANT: Reload best DGNO checkpoint before NF training
            best_dgno_path = self.weights_dir / 'best_dgno.pt'
            if best_dgno_path.exists():
                print(f"\nReloading best DGNO from: {best_dgno_path}")
                self.problem.load_checkpoint(
                    best_dgno_path,
                    models_to_load=['enc', 'u', 'a']  # Don't load NF
                )

            print("\n" + "=" * 60 + "\nPHASE 2: NF Training\n" + "=" * 60)
            results['nf'] = self._train_nf(train_data, test_data, config)

        print(f"\nCheckpoints saved to: {self.weights_dir}")
        return results

    def _train_dgno(self, train_data: Dict, test_data: Dict, config: TrainingConfig) -> Dict[str, Any]:
        """
        Train DGNO (encoder + decoders).

        NF is frozen during this phase.
        """
        cfg = config.dgno

        # Freeze NF, unfreeze DGNO models
        self.problem.freeze(['nf'])
        self.problem.unfreeze(['enc', 'u', 'a'])

        # Setup optimizer for DGNO models only
        dgno_params = []
        for name in ['enc', 'u', 'a']:
            dgno_params.extend(list(self.problem.model_dict[name].parameters()))

        self.optimizer = get_optimizer(cfg.optimizer, dgno_params)
        self.scheduler = get_scheduler(cfg.scheduler, self.optimizer)

        # Data loaders
        train_loader = data_loader(
            train_data['a'], train_data['u'], train_data['x'],
            cfg.batch_size, shuffle=True
        )
        test_loader = data_loader(
            test_data['a'], test_data['u'], test_data['x'],
            cfg.batch_size, shuffle=False
        )

        t_start = time.time()
        best_error = float('inf')
        weights = cfg.loss_weights

        for epoch in trange(cfg.epochs, desc="DGNO"):
            # Train mode for DGNO, eval for NF
            self.problem.train_mode(['enc', 'u', 'a'])
            self.problem.eval_mode(['nf'])

            loss_sum, pde_sum, data_sum = 0., 0., 0.

            for a, u, x in train_loader:
                a, u, x = a.to(self.device), u.to(self.device), x.to(self.device)

                loss_pde = self.problem.loss_pde(a)
                loss_data = self.problem.loss_data(x, a, u)
                loss = loss_pde * weights.pde + loss_data * weights.data

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dgno_params, max_norm=100.0)
                self.optimizer.step()

                loss_sum += loss.item()
                pde_sum += loss_pde.item()
                data_sum += loss_data.item()

            # Evaluation
            self.problem.eval_mode()
            error_sum = 0.

            with torch.no_grad():
                for a, u, x in test_loader:
                    a, u, x = a.to(self.device), u.to(self.device), x.to(self.device)
                    error_sum += self.problem.error(x, a, u).mean().item()

            n_train, n_test = len(train_loader), len(test_loader)
            avg_loss = loss_sum / n_train
            avg_error = error_sum / n_test

            # Logging
            self._log("dgno/loss", avg_loss, epoch)
            self._log("dgno/pde", pde_sum / n_train, epoch)
            self._log("dgno/data", data_sum / n_train, epoch)
            self._log("dgno/error", avg_error, epoch)

            # Save best
            if avg_error < best_error:
                best_error = avg_error
                self.problem.save_checkpoint(
                    self.weights_dir / 'best_dgno.pt',
                    epoch=epoch,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metric=avg_error,
                    metric_name='error'
                )

            # Scheduler step
            if self.scheduler:
                if cfg.scheduler.type == 'Plateau':
                    self.scheduler.step(avg_error)
                else:
                    self.scheduler.step()

            # Print progress
            if (epoch + 1) % cfg.epoch_show == 0:
                print(f"\nEpoch {epoch + 1}: Loss={avg_loss:.4f}, Error={avg_error:.4f}")

        # Save last checkpoint
        self.problem.save_checkpoint(
            self.weights_dir / 'last_dgno.pt',
            epoch=cfg.epochs - 1,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )

        return {'best_error': best_error, 'time': time.time() - t_start}

    def _train_nf(self, train_data: Dict, test_data: Dict, config: TrainingConfig) -> Dict[str, Any]:
        """
        Train Normalizing Flow on latent representations.
        DGNO models are frozen during this phase.
        """
        cfg = config.nf

        # Freeze DGNO, unfreeze NF
        self.problem.freeze(['enc', 'u', 'a'])
        self.problem.unfreeze(['nf'])

        # Setup optimizer for NF only
        nf = self.problem.model_dict['nf']
        self.optimizer = get_optimizer(cfg.optimizer, list(nf.parameters()))
        self.scheduler = get_scheduler(cfg.scheduler, self.optimizer)

        # Debug: print optimizer settings
        print(f"Optimizer: lr={self.optimizer.param_groups[0]['lr']}, "
              f"weight_decay={self.optimizer.param_groups[0]['weight_decay']}")

        # Extract latents from frozen encoder and keep on device
        print("Extracting latents...")
        latents_train = self._extract_latents(train_data['a'], cfg.batch_size).to(self.device)
        latents_test = self._extract_latents(test_data['a'], cfg.batch_size).to(self.device)

        print(f"Latents train: mean={latents_train.mean():.4f}, std={latents_train.std():.4f}")
        print(f"Latents test:  mean={latents_test.mean():.4f}, std={latents_test.std():.4f}")

        # =========== STANDARDIZATION ===========
        latent_mean = latents_train.mean(dim=0, keepdim=True)
        latent_std = latents_train.std(dim=0, keepdim=True) + 1e-8
        self.problem.set_latent_standardization(latent_mean, latent_std)

        latents_train = self.problem.standardize_latent(latents_train)
        latents_test = self.problem.standardize_latent(latents_test)

        print(f"Latents train (standardized): mean={latents_train.mean():.4f}, std={latents_train.std():.4f}")

        latents_train = latents_train.to(self.device)
        latents_test = latents_test.to(self.device)

        # =============================================

        print(f"Latents train (standardized): mean={latents_train.mean():.4f}, std={latents_train.std():.4f}")
        print(f"Latents test (standardized):  mean={latents_test.mean():.4f}, std={latents_test.std():.4f}")

        train_loader = var_data_loader(latents_train, batch_size=cfg.batch_size, shuffle=True)
        test_loader = var_data_loader(latents_test, batch_size=cfg.batch_size, shuffle=False)

        t_start = time.time()
        best_nll = float('inf')

        for epoch in trange(cfg.epochs, desc="NF"):
            # Train NF only
            self.problem.eval_mode(['enc', 'u', 'a'])
            self.problem.train_mode(['nf'])

            train_nll = 0.
            for (z,) in train_loader:
                loss = nf.loss(z)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=nf.parameters(), max_norm=5.0, error_if_nonfinite=True)
                self.optimizer.step()

                train_nll += loss.item()

            # Evaluation
            self.problem.eval_mode()
            test_nll = 0.

            with torch.no_grad():
                for (z,) in test_loader:
                    test_nll += nf.loss(z).item()

            avg_train = train_nll / len(train_loader)
            avg_test = test_nll / len(test_loader)

            # Logging
            self._log("nf/train_nll", avg_train, epoch)
            self._log("nf/test_nll", avg_test, epoch)

            # Save best
            if avg_test < best_nll:
                best_nll = avg_test
                self.problem.save_checkpoint(
                    self.weights_dir / 'best.pt',
                    epoch=epoch,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metric=avg_test,
                    metric_name='nll'
                )

            # Scheduler step
            if self.scheduler:
                if cfg.scheduler.type == 'Plateau':
                    self.scheduler.step(avg_test)
                else:
                    self.scheduler.step()

            # Debug prints every epoch_show
            if (epoch + 1) % cfg.epoch_show == 0:
                with torch.no_grad():
                    # Forward: latents -> z (should become N(0,1))
                    z_out, log_det_fwd = nf.forward(latents_train[:500])

                    # Inverse: z -> reconstructed latents (should match latent distribution)
                    z_sample = torch.randn(500, nf.dim, device=self.device)
                    beta_out, _ = nf.inverse(z_sample)

                    # Per-dimension statistics (catch collapse in specific dimensions)
                    z_std_per_dim = z_out.std(dim=0)
                    z_mean_per_dim = z_out.mean(dim=0)

                    # Check for dead/exploding dimensions
                    dead_dims = (z_std_per_dim < 0.1).sum().item()
                    exploding_dims = (z_std_per_dim > 3.0).sum().item()

                    # Log det statistics (should be reasonable, not exploding)
                    log_det_mean = log_det_fwd.mean().item()
                    log_det_std = log_det_fwd.std().item()

                    # Reconstruction test: latent -> z -> latent (should be identity)
                    z_roundtrip, _ = nf.forward(latents_train[:100])
                    beta_roundtrip, _ = nf.inverse(z_roundtrip)
                    reconstruction_err = (beta_roundtrip - latents_train[:100]).abs().mean().item()

                print(f"\nEpoch {epoch + 1}: Train NLL={avg_train:.4f}, Test NLL={avg_test:.4f}")
                print(f"  Forward (β→z):  mean={z_out.mean():.3f}, std={z_out.std():.3f} "
                      f"[target: mean=0, std=1]")
                print(f"  Per-dim z_std:  min={z_std_per_dim.min():.3f}, max={z_std_per_dim.max():.3f}, "
                      f"dead={dead_dims}, exploding={exploding_dims}")
                print(f"  Inverse (z→β):  mean={beta_out.mean():.4f}, std={beta_out.std():.4f} "
                      f"[target: mean≈0, std≈1 after standardization]")
                print(f"  Log-det:        mean={log_det_mean:.2f}, std={log_det_std:.2f}")
                print(f"  Reconstruction: {reconstruction_err:.6f} [should be ~0]")

        # Save last checkpoint
        self.problem.save_checkpoint(
            self.weights_dir / 'last.pt',
            epoch=cfg.epochs - 1,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )

        return {'best_nll': best_nll, 'time': time.time() - t_start}

    def _extract_latents(self, a: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Extract latent representations using frozen encoder."""
        enc = self.problem.model_dict['enc']
        enc.eval()

        latents = []
        with torch.no_grad():
            for i in range(0, len(a), batch_size):
                batch = a[i:i + batch_size].to(self.device)
                latents.append(enc(batch).cpu())

        return torch.cat(latents, dim=0)

    def close(self) -> None:
        """Cleanup resources."""
        if self.writer:
            self.writer.close()
            self.writer = None

