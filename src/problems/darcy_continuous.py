"""
Darcy Flow Continuous problem.

Original loss structure preserved:
- loss_pde(a): encodes a -> beta, then computes PDE loss
- loss_data(x, a, u): encodes a -> beta, then computes data loss

Additional _from_beta methods for inversion/encoder:
- loss_pde_from_beta(beta): PDE loss directly from beta
- loss_data_from_beta(beta, x, target, target_type): data loss directly from beta

All models (enc, u, a, nf) are built in _build_models().

Supports batched operations for efficient evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad, Variable
from typing import Dict, List
from pathlib import Path

from src.components.encoder import EncoderCNNet2dTanh
from src.components.nf import RealNVP
from src.problems import ProblemInstance, register_problem
from src.utils.GenPoints import Point2D
from src.utils.TestFun_ParticleWNN import TestFun_ParticleWNN
from src.utils.misc_utils import np2tensor
from src.utils.RBFInterpolatorMesh import RBFInterpolator
from src.utils.solver_utils import get_model
from src.utils.npy_loader import NpyFile

class TorchMollifier:
    def __call__(self, u, x):
        pi = torch.pi
        u = u * torch.sin(pi * x[..., 0]) * torch.sin(pi * x[..., 1])
        return u.unsqueeze(-1)

@register_problem("darcy_continuous")
class DarcyContinuous(ProblemInstance):
    """
    Continuous Darcy flow problem.

    PDE: -div(a * grad(u)) = f
    """

    # Hardcoded model hyperparameters (consistent across experiments)
    BETA_SIZE = 128
    HIDDEN_SIZE = 100
    NF_NUM_FLOWS = 3
    NF_HIDDEN_DIM = 64
    NF_NUM_LAYERS = 2

    def __init__(self, seed:int, device=None, dtype=torch.float32,
                 train_data_path: str = None, test_data_path: str = None):
        super().__init__(
            seed=seed,
            device=device,
            dtype=dtype,
            train_data_path=train_data_path,
            test_data_path=test_data_path,
        )

        # =====================================================================
        # 1. LOAD DATA
        # =====================================================================
        print("Loading data...")
        if self.train_data_path:
            self.train_data, self.gridx_train = self._load_data(self.train_data_path)
            print(f"  Train: a={self.train_data['a'].shape}, u={self.train_data['u'].shape}")
            self.fun_a = RBFInterpolator(
                x_mesh=self.gridx_train,
                kernel='gaussian',
                eps=25.,
                smoothing=0.,
                degree=6,
                dtype=self.dtype
            )

        if self.test_data_path:
            self.test_data, self.gridx_test = self._load_data(self.test_data_path)
            print(f"  Test:  a={self.test_data['a'].shape}, u={self.test_data['u'].shape}")

        # =====================================================================
        # 2. SETUP GRIDS & TEST FUNCTIONS
        # =====================================================================
        print("Setting up grids and test functions...")

        self.genPoint = Point2D(
            x_lb=[0., 0.],
            x_ub=[1., 1.],
            dataType=self.dtype,
            random_seed=self.seed
        )

        int_grid, v, dv_dr = TestFun_ParticleWNN(
            fun_type='Wendland',
            dim=2,
            n_mesh_or_grid=9,
            dataType=self.dtype
        ).get_testFun()

        self.int_grid = int_grid.to(self.device)
        self.v = v.to(self.device)
        self.dv_dr = dv_dr.to(self.device)
        self.n_grid = int_grid.shape[0]

        print(f"  int_grid: {self.int_grid.shape}, v: {self.v.shape}")

        self.mollifier = TorchMollifier()

        # =====================================================================
        # 3. BUILD MODELS (all models including NF)
        # =====================================================================
        print("Building models...")
        self.model_dict = self._build_models()

        for name, model in self.model_dict.items():
            model.to(self.device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  {name}: {n_params:,} parameters")

        print("Problem initialized.")

    def _load_data(self, path: str) -> tuple:
        """Load data from HDF5/MAT file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        data = NpyFile(path=path, mode='r')

        a = np2tensor(np.array(data["coeff"]).T, self.dtype)
        u = np2tensor(np.array(data["sol"]).T, self.dtype)

        X, Y = np.array(data['X']).T, np.array(data['Y']).T
        mesh = np2tensor(np.vstack([X.ravel(), Y.ravel()]).T, self.dtype)
        gridx = mesh.reshape(-1, 2)

        ndata = a.shape[0]
        a = a.reshape(ndata, -1, 1)
        x = gridx.repeat((ndata, 1, 1))
        u = u.reshape(ndata, -1, 1)

        return {'a': a, 'u': u, 'x': x}, gridx

    def _build_models(self) -> Dict[str, nn.Module]:
        """
        Build ALL models for Darcy flow problem.

        Returns:
            Dict with 'enc', 'u', 'a', 'nf'
        """
        net_type = 'MultiONetBatch'

        # ============== Encoder Architecture ==============
        conv_arch = [1, 64, 64, 64]
        fc_arch = [64 * 2 * 2, 128, 128, self.BETA_SIZE]
        model_enc = EncoderCNNet2dTanh(
            conv_arch=conv_arch,
            fc_arch=fc_arch,
            activation_conv='SiLU',
            activation_fc='SiLU',
            nx_size=29,
            ny_size=29,
            kernel_size=(3, 3),
            stride=2,
            dtype=self.dtype
        )

        # ============== Decoder Architecture ==============
        trunk_layers = [self.HIDDEN_SIZE] * 6
        branch_layers = [self.HIDDEN_SIZE] * 6

        model_a = get_model(
            x_in_size=2,
            beta_in_size=self.BETA_SIZE,
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk='Tanh_Sin',
            activation_branch='Tanh_Sin',
            net_type=net_type,
            sum_layers=5
        )

        model_u = get_model(
            x_in_size=2,
            beta_in_size=self.BETA_SIZE,
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk='Tanh_Sin',
            activation_branch='Tanh_Sin',
            net_type=net_type,
            sum_layers=5
        )

        # ============== Normalizing Flow ==============
        model_nf = RealNVP(
            dim=self.BETA_SIZE,
            num_flows=self.NF_NUM_FLOWS,
            hidden_dim=self.NF_HIDDEN_DIM,
            num_layers=self.NF_NUM_LAYERS,
        )

        return {
            'enc': model_enc,
            'u': model_u,
            'a': model_a,
            'nf': model_nf,
        }

    # =========================================================================
    # ORIGINAL LOSS METHODS (encode a -> beta first)
    # =========================================================================

    def loss_pde(self, a: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual loss."""
        nc = 100
        n_batch = a.shape[0]
        beta = self.model_dict['enc'](a)

        # Data points
        # xc:size(nc,1,2) R:size(nc,1,1)
        xc, R = self.genPoint.weight_centers(n_center=nc, R_max=1e-4, R_min=1e-4)
        xc, R = xc.to(self.device), R.to(self.device)

        x = self.int_grid * R + xc
        x = x.reshape(-1, 2).repeat((n_batch, 1, 1))
        x = Variable(x, requires_grad=True)

        v = self.v.repeat((nc, 1, 1)).reshape(-1, 1)
        dv = (self.dv_dr / R).reshape(-1, 2)

        a_detach = self.fun_a(x.detach(), a)
        u = self.model_dict['u'](x, beta)
        u = self.mollifier(u, x)

        du = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        f = 10.0 * torch.ones_like(u)

        left = torch.sum(a_detach * du * dv, dim=-1).reshape(n_batch, nc, self.n_grid)
        left = torch.mean(left, dim=-1)

        right = (f * v).reshape(n_batch, nc, self.n_grid)
        right = torch.mean(right, dim=-1)

        res = (left - right) ** 2
        res, indices = torch.sort(res.flatten(), descending=True, dim=0)
        loss_res = torch.sum(res[0:nc * 10])

        return self.get_loss(left, right) + loss_res

    def loss_data(self, x: torch.Tensor, a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute data loss. First encodes a to beta, then computes loss.

        Args:
            x: Coordinates (batch, n_points, 2)
            a: Coefficient field (batch, n_points, 1) - used for encoding
            u: Solution field (batch, n_points, 1) - target (NOT USED in original)

        Returns:
            Data fitting loss (on coefficient a)
        """
        beta = self.model_dict['enc'](a)
        return self.loss_data_from_beta(
            beta=beta,
            x=x,
            target=a,
            target_type='a'
        )

    def error(self, x: torch.Tensor, a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute error metric. First encodes a to beta, then computes error on u.

        Args:
            x: Coordinates (batch, n_points, 2)
            a: Coefficient field (batch, n_points, 1) - used for encoding
            u: Solution field (batch, n_points, 1) - target for error

        Returns:
            Relative L2 error on solution u
        """
        beta = self.model_dict['enc'](a)
        return self.error_from_beta(
            beta=beta,
            x=x,
            target=u,
            target_type='u',
        )

    # =========================================================================
    # FROM_BETA METHODS (for inversion/encoder - skip encoding step)
    # Supports batched beta inputs for efficient evaluation.
    # =========================================================================

    def loss_pde_from_beta(self, beta: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual loss given beta directly.

        -div(a * grad(u)) = f, with f = 10

        Supports batched inputs for efficient parallel inversion.

        Args:
            beta: Latent representation (batch, latent_dim)

        Returns:
            PDE residual loss (scalar mean over batch)
        """
        nc = 100
        n_batch = beta.shape[0]

        # Data points (shared across all samples in batch)
        # xc:size(nc, 1, 2) R:size(nc, 1, 1)
        xc, R = self.genPoint.weight_centers(n_center=nc, R_max=1e-4, R_min=1e-4)
        xc, R = xc.to(self.device), R.to(self.device)
        # size(nc, n_grid, 2)
        x = self.int_grid * R + xc
        # size(nc*n_grid, 2) -> (n_batch, nc*n_grid, 2)
        x = x.reshape(-1, 2).repeat((n_batch, 1, 1))
        x = Variable(x, requires_grad=True)

        # Test functions
        v = self.v.repeat((nc, 1, 1)).reshape(-1, 1)
        dv = (self.dv_dr / R).reshape(-1, 2)

        # Model prediction
        a_detach = self.model_dict['a'](x.detach(), beta)
        a_detach = a_detach.unsqueeze(-1)
        # u: size(n_batch, nc*n_grid, 1)
        u = self.model_dict['u'](x, beta)
        u = self.mollifier(u, x)
        # du: size(n_batch, nc*n_grid, 2)
        du = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        f = 10. * torch.ones_like(u)

        # PDE loss
        # size(n_batch, nc*n_grid, 2) -> (n_batch, nc, n_grid) -> (n_batch, nc)
        left = torch.sum(a_detach * (du * dv), dim=-1).reshape(n_batch, nc, self.n_grid)
        left = torch.mean(left, dim=-1)
        # size(n_batch, nc*n_grid, 1) -> (n_batch, nc, n_grid) -> (n_batch, nc)
        right = (f * v).reshape(n_batch, nc, self.n_grid)
        right = torch.mean(right, dim=-1)

        loss_pde = torch.norm(left - right, 2, 1)  # (n_batch,)

        return torch.mean(loss_pde)

    def loss_data_from_beta(self, beta: torch.Tensor, x: torch.Tensor,
                            target: torch.Tensor, target_type: str = 'a') -> torch.Tensor:
        """
        Compute data fitting loss given beta directly.

        Supports batched inputs for efficient parallel inversion.

        Args:
            beta: Latent representation (batch, latent_dim)
            x: Coordinates (batch, n_points, 2)
            target: Target values (batch, n_points, 1)
            target_type: 'a' for coefficient, 'u' for solution

        Returns:
            Data fitting loss (scalar mean over batch)
        """
        if target_type == 'a':
            pred = self.model_dict['a'](x, beta)
            return self.get_loss(pred, target.squeeze(-1))
        elif target_type == 'u':
            # I debugged this earlier, both pred and target are [batch_size, obs_size, 1]
            pred = self.model_dict['u'](x, beta)
            pred = self.mollifier(pred, x)
            loss = torch.norm(pred - target, 2, 1) / torch.norm(target, 2, 1)  # RELATIVE!
            return torch.mean(loss)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    def error_from_beta(self, beta: torch.Tensor, x: torch.Tensor,
                        target: torch.Tensor, target_type: str = 'u') -> torch.Tensor:
        """
        Compute error metric given beta directly.

        Args:
            beta: Latent representation (batch, latent_dim)
            x: Coordinates (batch, n_points, 2)
            target: Target values
            target_type: 'u' for solution, 'a' for coefficient

        Returns:
            Relative Lp error
        """
        if target_type == 'u':
            pred = self.model_dict['u'](x, beta)
            pred = self.mollifier(pred, x)
        elif target_type == 'a':
            pred = self.model_dict['a'](x, beta)
            if target.dim() == 3 and target.shape[-1] == 1:
                target = target.squeeze(-1)
            pred = pred.unsqueeze(-1) if pred.dim() == 2 else pred
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        return self.get_error(pred, target)

    # =========================================================================
    # PREDICTION METHODS
    # =========================================================================

    def predict_from_beta(self, beta: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Given beta, predict u and a on given coordinates.

        Supports batched inputs.

        Args:
            beta: Latent representation (batch, latent_dim)
            x: Coordinates (batch, n_points, 2)

        Returns:
            {'u_pred': Tensor, 'a_pred': Tensor}
        """
        with torch.no_grad():
            u_pred = self.model_dict['u'](x, beta)
            u_pred = self.mollifier(u_pred, x)
            a_pred = self.model_dict['a'](x, beta)
            a_pred = a_pred.unsqueeze(-1)

        # I debugged this, Both are [batch_size, output_grid_size, 1] after the transformations

        return {
            'u_pred': u_pred,
            'a_pred':  a_pred,
        }

    # =========================================================================
    # OBSERVATION METHODS (for evaluation/inversion)
    # =========================================================================

    def prepare_observations(
        self,
        sample_indices: List[int],
        obs_indices: np.ndarray,
        snr_db: float = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare observation data for test samples.

        Supports batched preparation for efficient evaluation.

        Args:
            sample_indices: List of indices into test set
            obs_indices: Indices of observation points (same for all samples)
            snr_db: SNR for noise (None for clean)

        Returns:
            Dict with x_full, x_obs, u_obs, u_true, a_true
            All tensors have shape (batch, ...)
        """
        batch_size = len(sample_indices)

        # Stack all samples along batch dimension
        a_true = torch.stack([self.test_data['a'][i] for i in sample_indices])  # (batch, n_points, 1)
        u_true = torch.stack([self.test_data['u'][i] for i in sample_indices])  # (batch, n_points, 1)
        x_full = torch.stack([self.test_data['x'][i] for i in sample_indices])  # (batch, n_points, 2)

        # Extract observations (same indices for all samples)
        x_obs = x_full[:, obs_indices, :]      # (batch, n_obs, 2)
        u_obs = u_true[:, obs_indices, :].clone()  # (batch, n_obs, 1)

        # Add noise if specified
        if snr_db is not None:
            u_obs = self.add_noise_snr(u_obs, snr_db)

        # print(f"x_full.shape: {x_full.shape}, x_obs.shape: {x_obs.shape}, u_obs.shape: {u_obs.shape}, u_true.shape: {u_true.shape}, a_true.shape: {a_true.shape}")

        return {
            'x_full': x_full.to(self.device),      # (batch, n_points, 2)
            'x_obs': x_obs.to(self.device),         # (batch, n_obs, 2)
            'u_obs': u_obs.to(self.device),         # (batch, n_obs, 1)
            'u_true': u_true.to(self.device),       # (batch, n_points, 1)
            'a_true': a_true.to(self.device),       # (batch, n_points, 1)
        }

    def get_n_test_samples(self) -> int:
        """Get number of test samples."""
        return len(self.test_data['a'])

    def get_n_points(self) -> int:
        """Get number of grid points per sample."""
        return self.test_data['x'].shape[1]
