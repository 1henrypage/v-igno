"""
Darcy Flow Continuous problem.

Original loss structure preserved:
- loss_pde(a): encodes a -> beta, then computes PDE loss
- loss_data(x, a, u): encodes a -> beta, then computes data loss

Additional _from_beta methods for inversion/encoder:
- loss_pde_from_beta(beta): PDE loss directly from beta
- loss_data_from_beta(beta, x, target, target_type): data loss directly from beta
"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad, Variable
from typing import Dict
from pathlib import Path

from src.problems import ProblemInstance, register_problem
from src.utils.GenPoints import Point2D
from src.utils.TestFun_ParticleWNN import TestFun_ParticleWNN
from src.utils.misc_utils import np2tensor
from src.components.activation import Mollifier
from utils.npy_loader import NpyFile


@register_problem("darcy_flow_continuous")
class DarcyFlowContinuous(ProblemInstance):
    """
    Continuous Darcy flow problem.

    PDE: -div(a * grad(u)) = f
    """

    # =========================================================================
    # PROBLEM PARAMETERS
    # =========================================================================
    NC = 100              # number of collocation centers for PDE loss
    N_MESH = 9            # mesh size for test functions
    R_MAX = 1e-4
    R_MIN = 1e-4
    X_LB = [0., 0.]
    X_UB = [1., 1.]

    def __init__(self, device=None, dtype=torch.float32, seed: int = 10086,
                 train_data_path: str = None, test_data_path: str = None):
        super().__init__(
            device=device,
            dtype=dtype,
            seed=seed,
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

        if self.test_data_path:
            self.test_data, self.gridx_test = self._load_data(self.test_data_path)
            print(f"  Test:  a={self.test_data['a'].shape}, u={self.test_data['u'].shape}")

        # =====================================================================
        # 2. SETUP GRIDS & TEST FUNCTIONS
        # =====================================================================
        print("Setting up grids and test functions...")

        self.genPoint = Point2D(
            x_lb=self.X_LB,
            x_ub=self.X_UB,
            dataType=self.dtype
        )

        int_grid, v, dv_dr = TestFun_ParticleWNN(
            fun_type='Wendland',
            dim=2,
            n_mesh_or_grid=self.N_MESH,
            dataType=self.dtype
        ).get_testFun()

        self.int_grid = int_grid.to(self.device)
        self.v = v.to(self.device)
        self.dv_dr = dv_dr.to(self.device)
        self.n_grid = int_grid.shape[0]

        print(f"  int_grid: {self.int_grid.shape}, v: {self.v.shape}")

        self.mollifier = Mollifier()

        # =====================================================================
        # 3. BUILD MODELS (only for training, not evaluation)
        # =====================================================================
        # For evaluation, models are set via set_models() after loading checkpoint
        if self.train_data_path:
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
        Build encoder, u-decoder, and a-decoder.

        TODO: Implement with your model architectures.
        """
        raise NotImplementedError("Implement _build_models() with your architectures")

    # =========================================================================
    # ORIGINAL LOSS METHODS (encode a -> beta first)
    # =========================================================================

    def loss_pde(self, a: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE loss. First encodes a to beta, then computes residual.

        Args:
            a: Coefficient field (batch, n_points, 1)

        Returns:
            PDE residual loss
        """
        beta = self.model_dict['enc'](a)
        return self.loss_pde_from_beta(beta)

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
        # Original trains to reconstruct 'a', not 'u'
        return self.loss_data_from_beta(beta, x, a, target_type='a')

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
        return self.error_from_beta(beta, x, u, target_type='u')

    # =========================================================================
    # FROM_BETA METHODS (for inversion/encoder - skip encoding step)
    # =========================================================================

    def loss_pde_from_beta(self, beta: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual loss given beta directly.

        -div(a * grad(u)) = f, with f = 10

        Args:
            beta: Latent representation (batch, latent_dim)

        Returns:
            PDE residual loss
        """
        nc = self.NC
        n_batch = beta.shape[0]

        # Generate collocation points
        xc, R = self.genPoint.weight_centers(n_center=nc, R_max=self.R_MAX, R_min=self.R_MIN)
        xc, R = xc.to(self.device), R.to(self.device)

        x = self.int_grid * R + xc
        x = x.reshape(-1, 2).repeat((n_batch, 1, 1))
        x = Variable(x, requires_grad=True)

        # Test functions
        v = self.v.repeat((nc, 1, 1)).reshape(-1, 1)
        dv = (self.dv_dr / R).reshape(-1, 2)

        # Model predictions using beta
        a_pred = self.model_dict['a'](x.detach(), beta)
        a_pred = a_pred.unsqueeze(-1)

        u = self.model_dict['u'](x, beta)
        u = self.mollifier(u, x)

        du = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        f = 10.0 * torch.ones_like(u)

        # PDE residual: int (a * grad(u) * grad(v)) dx = int (f * v) dx
        left = torch.sum(a_pred * du * dv, dim=-1).reshape(n_batch, nc, self.n_grid)
        left = torch.mean(left, dim=-1)

        right = (f * v).reshape(n_batch, nc, self.n_grid)
        right = torch.mean(right, dim=-1)

        # Top-k residual loss
        res = (left - right) ** 2
        res, _ = torch.sort(res.flatten(), descending=True, dim=0)
        loss_res = torch.sum(res[0:nc * 10])

        return self.get_loss(left, right) + loss_res

    def loss_data_from_beta(self, beta: torch.Tensor, x: torch.Tensor,
                            target: torch.Tensor, target_type: str = 'a') -> torch.Tensor:
        """
        Compute data fitting loss given beta directly.

        Args:
            beta: Latent representation (batch, latent_dim)
            x: Coordinates (batch, n_points, 2)
            target: Target values (batch, n_points, 1)
            target_type: 'a' for coefficient, 'u' for solution

        Returns:
            Data fitting loss
        """
        if target_type == 'a':
            pred = self.model_dict['a'](x, beta)
            return self.get_loss(pred, target.squeeze(-1))
        elif target_type == 'u':
            pred = self.model_dict['u'](x, beta)
            pred = self.mollifier(pred, x)
            # Relative L2 norm (matching their code)
            loss = torch.norm(pred - target, 2, 1) / torch.norm(target, 2, 1)
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

        return {
            'u_pred': u_pred,
            'a_pred': a_pred.unsqueeze(-1) if a_pred.dim() == 2 else a_pred,
        }

    # =========================================================================
    # OBSERVATION METHODS (for evaluation/inversion)
    # =========================================================================

    def prepare_observations(
        self,
        sample_idx: int,
        obs_indices: np.ndarray,
        snr_db: float = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare observation data for a single test sample.

        Args:
            sample_idx: Index into test set
            obs_indices: Indices of observation points
            snr_db: SNR for noise (None for clean)

        Returns:
            Dict with x_full, x_obs, u_obs, u_true, a_true
        """
        # Get single sample from test set
        a_true = self.test_data['a'][sample_idx]  # (n_points, 1)
        u_true = self.test_data['u'][sample_idx]  # (n_points, 1)
        x_full = self.test_data['x'][sample_idx]  # (n_points, 2)

        # Extract observations
        x_obs = x_full[obs_indices]
        u_obs = u_true[obs_indices].clone()

        # Add noise if specified
        if snr_db is not None:
            u_obs = self.add_noise_snr(u_obs, snr_db)

        return {
            'x_full': x_full.unsqueeze(0).to(self.device),      # (1, n_points, 2)
            'x_obs': x_obs.unsqueeze(0).to(self.device),         # (1, n_obs, 2)
            'u_obs': u_obs.unsqueeze(0).to(self.device),         # (1, n_obs, 1)
            'u_true': u_true.unsqueeze(0).to(self.device),       # (1, n_points, 1)
            'a_true': a_true.unsqueeze(0).to(self.device),       # (1, n_points, 1)
        }

    def get_n_test_samples(self) -> int:
        """Get number of test samples."""
        return len(self.test_data['a'])

    def get_n_points(self) -> int:
        """Get number of grid points per sample."""
        return self.test_data['x'].shape[1]