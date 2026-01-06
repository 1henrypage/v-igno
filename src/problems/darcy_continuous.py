"""
Darcy Flow Continuous problem.

Everything in constructor:
- Data loading
- Grid/test function setup
- Model creation
"""
import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.autograd import grad, Variable
from typing import Dict
from pathlib import Path

from src.problems import ProblemInstance, register_problem
from src.utils.GenPoints import Point2D
from src.utils.TestFun_ParticleWNN import TestFun_ParticleWNN
from src.utils.misc_utils import np2tensor


class TorchMollifier:
    def __call__(self, u, x):
        pi = torch.pi
        u = u * torch.sin(pi * x[..., 0]) * torch.sin(pi * x[..., 1])
        return u.unsqueeze(-1)


@register_problem("darcy_flow_continuous")
class DarcyFlowContinuous(ProblemInstance):
    """
    Continuous Darcy flow problem.
    All configuration here in Python.
    """

    # =========================================================================
    # CONFIGURATION - Edit these directly
    # =========================================================================

    # Data paths
    TRAIN_DATA_PATH = "data/darcy_continuous/smh_train.mat"
    TEST_DATA_PATH = "data/darcy_continuous/smh_test_in.mat"

    def __init__(self, device=None, dtype=torch.float32, seed: int = 10086):
        super().__init__(device=device, dtype=dtype, seed=seed)

        # =====================================================================
        # 1. LOAD DATA
        # =====================================================================
        print("Loading data...")
        self.train_data, self.gridx_train = self._load_data(self.TRAIN_DATA_PATH)
        self.test_data, self.gridx_test = self._load_data(self.TEST_DATA_PATH)

        print(f"  Train: a={self.train_data['a'].shape}, u={self.train_data['u'].shape}")
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
        # 3. BUILD MODELS
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

        data = h5py.File(path, 'r')

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
        Build encoder, u-model, and a-model.

        TODO: Implement with your model architectures.
        """
        # Example:
        # model_enc = YourEncoder(...)
        # model_u = YourUModel(...)
        # model_a = YourAModel(...)
        # return {'enc': model_enc, 'u': model_u, 'a': model_a}

        raise NotImplementedError("Implement _build_models() with your architectures")

    def _fun_a(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Evaluate coefficient function a at points x."""
        raise NotImplementedError("Implement _fun_a()")

    def loss_pde(self, a: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual loss."""
        nc = self.NC
        n_batch = a.shape[0]

        beta = self.model_dict['enc'](a)

        xc, R = self.genPoint.weight_centers(n_center=nc, R_max=self.R_MAX, R_min=self.R_MIN)
        xc, R = xc.to(self.device), R.to(self.device)

        x = self.int_grid * R + xc
        x = x.reshape(-1, 2).repeat((n_batch, 1, 1))
        x = Variable(x, requires_grad=True)

        v = self.v.repeat((nc, 1, 1)).reshape(-1, 1)
        dv = (self.dv_dr / R).reshape(-1, 2)

        a_detach = self._fun_a(x.detach(), a)
        u = self.model_dict['u'](x, beta)
        u = self.mollifier(u, x)

        du = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        f = 10.0 * torch.ones_like(u)

        left = torch.sum(a_detach * du * dv, dim=-1).reshape(n_batch, nc, self.n_grid)
        left = torch.mean(left, dim=-1)

        right = (f * v).reshape(n_batch, nc, self.n_grid)
        right = torch.mean(right, dim=-1)

        res = (left - right) ** 2
        res, _ = torch.sort(res.flatten(), descending=True, dim=0)
        loss_res = torch.sum(res[0:nc * 10])

        return self.get_loss(left, right) + loss_res

    def loss_data(self, x: torch.Tensor, a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute data fitting loss on a."""
        beta = self.model_dict['enc'](a)
        a_pred = self.model_dict['a'](x, beta)
        return self.get_loss(a_pred, a.squeeze(-1))

    def error(self, x: torch.Tensor, a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute prediction error on u."""
        beta = self.model_dict['enc'](a)
        u_pred = self.model_dict['u'](x, beta)
        u_pred = self.mollifier(u_pred, x)
        return self.get_error(u_pred, u)