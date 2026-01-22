"""
IGNO Trainer: Joint training of encoder + decoders + NF.

This trainer implements the IGNO training procedure where the normalizing flow
is trained jointly with the encoder and decoders. The NF learns the latent
distribution using DETACHED latents (gradients don't flow back to encoder).

Training loss:
    L = w_pde * L_pde(a) + w_data * L_data(x, a, u) + L_nf(beta.detach())

Where:
- L_pde(a) encodes a -> beta, then computes PDE loss using TRUE coefficient via RBF
- L_data(x, a, u) encodes a -> beta, then computes reconstruction loss on a
- L_nf = -mean(log p(beta)) computed on beta.detach()
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
from src.utils.solver_utils import get_optimizer, get_scheduler, data_loader


class IGNOTrainer:
    """
    Joint trainer for IGNO: encoder + decoders + normalizing flow.

    All models are owned by the ProblemInstance - this class orchestrates training.

    The key insight from the original IGNO implementation is that the NF is trained
    jointly using detached latents:
        z, prior_logprob, log_det = model_NF(beta.detach())
        loss_NF = -mean(prior_logprob + log_det)

    This allows the NF to learn the latent distribution without affecting encoder gradients.
    """

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
        self.weights_dir: Optional[Path] = None
        self.tb_dir: Optional[Path] = None

    def _create_run_dir(self, config: TrainingConfig) -> Path:
        """Create timestamped run directory."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_dir = Path(config.artifact_root) / f"{timestamp}_{config.run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _setup_directories(self) -> None:
        """Setup directories for weights and tensorboard."""
        if self.run_dir is None:
            raise RuntimeError("run_dir must be set before calling _setup_directories")

        self.weights_dir = self.run_dir / "weights"
        self.tb_dir = self.run_dir / "tensorboard"

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

        # Store standardization setting in problem for checkpoint saving
        self.problem.standardize_latent_enabled = config.training.standardize_latent

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

    def train(self, config: TrainingConfig) -> Dict[str, Any]:
        """
        Train IGNO: encoder + decoders + NF jointly.

        The training follows the original IGNO paper:
        1. Encode coefficient field a -> beta (inside loss_pde and loss_data)
        2. Compute PDE loss using TRUE coefficient via RBF interpolator
        3. Compute reconstruction loss on coefficient a
        4. Compute NF loss on beta.detach() (gradients don't flow to encoder)
        5. Update all models together

        Args:
            config: Training configuration

        Returns:
            Results dict with metrics
        """
        cfg = config.training
        train_data = self.problem.get_train_data()
        test_data = self.problem.get_test_data()

        param_buckets = {"decay": [], "no_decay": []}

        for name, model in self.problem.model_dict.items():
            model.train()
            if name in ["a", "u"]:
                param_buckets["decay"].extend(list(model.parameters()))
            else:
                param_buckets["no_decay"].extend(list(model.parameters()))


        self.optimizer = get_optimizer(cfg.optimizer, param_buckets)
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

        # If using standardization, compute stats from initial encoder pass
        if cfg.standardize_latent:
            self._compute_latent_statistics(train_data['a'], cfg.batch_size)

        t_start = time.time()
        best_error = float('inf')
        weights = cfg.loss_weights

        print("\n" + "=" * 60)
        print("IGNO Joint Training (encoder + decoders + NF)")
        print("=" * 60)
        print(f"Standardize latent: {cfg.standardize_latent}")
        print(f"Loss weights: pde={weights.pde}, data={weights.data}")
        print(f"Epochs: {cfg.epochs}, Batch size: {cfg.batch_size}")
        print("=" * 60 + "\n")

        for epoch in trange(cfg.epochs, desc="Training"):
            # Train mode
            self.problem.train_mode()

            loss_sum = 0.
            pde_sum = 0.
            data_sum = 0.
            nf_sum = 0.

            for a, u, x in train_loader:
                a, u, x = a.to(self.device), u.to(self.device), x.to(self.device)

                # Get beta for NF loss (need to encode manually to get beta for NF)
                beta = self.problem.model_dict['enc'](a)

                # PDE loss - uses loss_pde(a) which encodes internally and uses RBF interpolator
                loss_pde = self.problem.loss_pde(a)

                # Data/reconstruction loss - uses loss_data which encodes internally
                loss_data = self.problem.loss_data(x, a, u)

                # NF loss on DETACHED beta (key insight from author)
                # This trains NF to model the latent distribution without affecting encoder
                loss_nf = self._compute_nf_loss(beta.detach(), cfg.standardize_latent)

                # Total loss
                loss = weights.pde * loss_pde + weights.data * loss_data + loss_nf

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(param_buckets["decay"] + param_buckets["no_decay"], max_norm=10.0, error_if_nonfinite=True)
                self.optimizer.step()

                loss_sum += loss.item()
                pde_sum += loss_pde.item()
                data_sum += loss_data.item()
                nf_sum += loss_nf.item()

            # Update latent statistics periodically if using standardization
            if cfg.standardize_latent and (epoch + 1) % 100 == 0:
                self._compute_latent_statistics(train_data['a'], cfg.batch_size)

            # Evaluation
            self.problem.eval_mode()
            error_sum = 0.
            test_nf_sum = 0.

            with torch.no_grad():
                for a, u, x in test_loader:
                    a, u, x = a.to(self.device), u.to(self.device), x.to(self.device)
                    error_sum += self.problem.error(x, a, u).mean().item()

                    # Also track NF loss on test set
                    beta_test = self.problem.model_dict['enc'](a)
                    test_nf_sum += self._compute_nf_loss(beta_test, cfg.standardize_latent).item()

            n_train, n_test = len(train_loader), len(test_loader)
            avg_loss = loss_sum / n_train
            avg_error = error_sum / n_test
            avg_nf = nf_sum / n_train
            avg_test_nf = test_nf_sum / n_test

            # Logging
            self._log("train/loss", avg_loss, epoch)
            self._log("train/pde", pde_sum / n_train, epoch)
            self._log("train/data", data_sum / n_train, epoch)
            self._log("train/nf", avg_nf, epoch)
            self._log("test/error", avg_error, epoch)
            self._log("test/nf", avg_test_nf, epoch)

            # Save best based on reconstruction error
            if avg_error < best_error:
                best_error = avg_error
                self.problem.save_checkpoint(
                    self.weights_dir / 'best.pt',
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
                print(f"\nEpoch {epoch + 1}:")
                print(f"  Loss: {avg_loss:.4f} (pde={pde_sum/n_train:.4f}, "
                      f"data={data_sum/n_train:.4f}, nf={avg_nf:.4f})")
                print(f"  Test Error: {avg_error:.4f}, Test NF NLL: {avg_test_nf:.4f}")
                # --- NF Observability Block ---
                self._run_nf_diagnostics(epoch, train_data['a'], cfg.standardize_latent)

        # Save last checkpoint
        self.problem.save_checkpoint(
            self.weights_dir / 'last.pt',
            epoch=cfg.epochs - 1,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )

        total_time = time.time() - t_start
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Best error: {best_error:.4f}")
        print(f"Checkpoints saved to: {self.weights_dir}")

        return {
            'best_error': best_error,
            'time': total_time,
        }

    def _compute_nf_loss(self, beta: torch.Tensor, standardize: bool) -> torch.Tensor:
        """
        Compute NF negative log-likelihood loss.

        Args:
            beta: Latent representations (already detached)
            standardize: Whether to standardize before NF

        Returns:
            NF loss (negative mean log-prob)
        """
        nf = self.problem.model_dict['nf']

        if standardize:
            beta = self.problem.standardize_latent(beta)

        return nf.loss(beta)

    def _compute_latent_statistics(self, a: torch.Tensor, batch_size: int) -> None:
        """
        Compute latent mean and std from encoder for standardization.

        Args:
            a: All training coefficient fields
            batch_size: Batch size for encoding
        """
        enc = self.problem.model_dict['enc']
        enc.eval()

        latents = []
        with torch.no_grad():
            for i in range(0, len(a), batch_size):
                batch = a[i:i + batch_size].to(self.device)
                latents.append(enc(batch))

        latents = torch.cat(latents, dim=0)
        mean = latents.mean(dim=0, keepdim=True)
        std = latents.std(dim=0, keepdim=True) + 1e-8

        self.problem.set_latent_standardization(mean, std)
        enc.train()

    def _run_nf_diagnostics(self, epoch: int, a_sample: torch.Tensor, standardize: bool) -> None:
        """Detailed monitoring of NF health and expressivity."""
        nf = self.problem.model_dict['nf']
        enc = self.problem.model_dict['enc']

        nf.eval()
        enc.eval()

        with torch.no_grad():
            # 1. Get Latents (Standardized if enabled)
            beta = enc(a_sample.to(self.device))
            if standardize:
                beta = self.problem.standardize_latent(beta)

            # 2. Forward: Latent -> Z (Target: N(0,1))
            z_out, log_det_fwd = nf.forward(beta)

            # 3. Stats for Z-space (Detecting Mode Collapse)
            z_mean_per_dim = z_out.mean(dim=0)
            z_std_per_dim = z_out.std(dim=0)

            z_mean_total = z_out.mean().item()
            z_std_total = z_out.std().item()
            dead_dims = (z_std_per_dim < 0.1).sum().item()
            exploding_dims = (z_std_per_dim > 5.0).sum().item()

            # 4. Invertibility Check (Roundtrip: beta -> z -> beta_rec)
            beta_rec, _ = nf.inverse(z_out)
            rec_err = torch.abs(beta - beta_rec).mean().item()

            # 5. Log-Det Analysis (Numerical stability)
            ldj_mean = log_det_fwd.mean().item()
            ldj_std = log_det_fwd.std().item()

            # --- Logging to TensorBoard ---
            self._log("nf_health/z_mean_avg", z_mean_total, epoch)
            self._log("nf_health/z_std_avg", z_std_total, epoch)
            self._log("nf_health/dead_dims", float(dead_dims), epoch)
            self._log("nf_health/rec_error_abs", rec_err, epoch)
            self._log("nf_health/log_det_mean", ldj_mean, epoch)

            # --- Console Output ---
            print(f"  [NF Health] Roundtrip Err: {rec_err:.2e} | Dead Dims: {dead_dims}/{z_out.shape[1]}")
            print(f"  [NF Health] Z-Space: Mean={z_mean_total:.3f}, Std={z_std_total:.3f} | LogDet: {ldj_mean:.2f}")

            if dead_dims > (z_out.shape[1] // 2):
                print("  ⚠️ WARNING: High number of dead dimensions detected. Flow might be collapsing.")
            if rec_err > 1e-3:
                print("  ⚠️ WARNING: Poor invertibility. Check for vanishing/exploding gradients in NF.")

        nf.train()
        enc.train()

    def close(self) -> None:
        """Cleanup resources."""
        if self.writer:
            self.writer.close()
            self.writer = None
