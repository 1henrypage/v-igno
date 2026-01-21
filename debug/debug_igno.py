#!/usr/bin/env python3
"""
Standalone debug script for IGNO inversion.
Manually loads models to avoid circular import issues.

Usage:
    python debug_igno_standalone.py --checkpoint /path/to/best.pt --data data/darcy_continuous/smh_test_in/
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.autograd import grad, Variable


# =============================================================================
# Minimal model definitions (copied from your codebase)
# =============================================================================

class FunActivation:
    def __call__(self, name):
        if name == 'Tanh_Sin':
            return lambda x: torch.tanh(torch.sin(np.pi * x + np.pi)) + x
        elif name == 'SiLU':
            return nn.SiLU()
        elif name == 'Tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {name}")


class MultiONetBatch(nn.Module):
    def __init__(self, in_size_x, in_size_a, trunk_layers, branch_layers,
                 activation_trunk='SiLU_Sin', activation_branch='SiLU',
                 sum_layers=4, dtype=None):
        super().__init__()
        self.l = sum_layers

        self.activation_trunk = FunActivation()(activation_trunk)
        self.activation_branch = FunActivation()(activation_branch)

        self.fc_trunk_in = nn.Linear(in_size_x, trunk_layers[0], dtype=dtype)
        trunk_net = []
        hidden_in = trunk_layers[0]
        for hidden in trunk_layers[1:]:
            trunk_net.append(nn.Linear(hidden_in, hidden, dtype=dtype))
            hidden_in = hidden
        self.trunk_net = nn.Sequential(*trunk_net)

        self.fc_branch_in = nn.Linear(in_size_a, branch_layers[0], dtype=dtype)
        branch_net = []
        hidden_in = branch_layers[0]
        for hidden in branch_layers[1:]:
            branch_net.append(nn.Linear(hidden_in, hidden, dtype=dtype))
            hidden_in = hidden
        self.branch_net = nn.Sequential(*branch_net)

        self.w = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(self.l)])
        self.b = nn.Parameter(torch.tensor(0.0, dtype=dtype))

    def forward(self, x, a):
        x = self.activation_trunk(self.fc_trunk_in(x))
        a = self.activation_branch(self.fc_branch_in(a))

        for net_t, net_b in zip(self.trunk_net[:-self.l], self.branch_net[:-self.l]):
            x = self.activation_trunk(net_t(x))
            a = self.activation_branch(net_b(a))

        out = 0.
        for net_t, net_b, w in zip(self.trunk_net[-self.l:], self.branch_net[-self.l:], self.w):
            x = self.activation_trunk(net_t(x))
            a = self.activation_branch(net_b(a))
            out += torch.einsum('bnh,bh->bn', x, a) * w

        return out / self.l + self.b


class EncoderCNNet2dTanh(nn.Module):
    def __init__(self, conv_arch, fc_arch, activation_conv='SiLU', activation_fc='SiLU',
                 nx_size=29, ny_size=29, kernel_size=(3, 3), stride=2, dtype=None):
        super().__init__()

        if activation_conv == 'SiLU':
            self.act_conv = nn.SiLU()
        else:
            self.act_conv = nn.ReLU()

        if activation_fc == 'SiLU':
            self.act_fc = nn.SiLU()
        else:
            self.act_fc = nn.ReLU()

        conv_layers = []
        in_channels = conv_arch[0]
        for out_channels in conv_arch[1:]:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0))
            conv_layers.append(self.act_conv)
            in_channels = out_channels
        self.conv_net = nn.Sequential(*conv_layers)

        fc_layers = []
        in_features = fc_arch[0]
        for out_features in fc_arch[1:-1]:
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(self.act_fc)
            in_features = out_features
        fc_layers.append(nn.Linear(in_features, fc_arch[-1]))
        fc_layers.append(nn.Tanh())
        self.fc_net = nn.Sequential(*fc_layers)

        self.nx = nx_size
        self.ny = ny_size

    def forward(self, a):
        # a: (batch, n_points, 1) -> reshape to image
        batch = a.shape[0]
        a_img = a.squeeze(-1).reshape(batch, 1, self.nx, self.ny)

        h = self.conv_net(a_img)
        h = h.reshape(batch, -1)
        return self.fc_net(h)


class ScaleTranslateNet(nn.Module):
    def __init__(self, cond_dim, out_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        layers = [nn.Linear(cond_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        self.net = nn.Sequential(*layers)
        self.scale_layer = nn.Linear(hidden_dim, out_dim)
        self.translate_layer = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.scale_layer.weight)
        nn.init.zeros_(self.scale_layer.bias)

    def forward(self, x):
        h = self.net(x)
        scale = torch.tanh(self.scale_layer(h)) * 2.0
        translation = self.translate_layer(h)
        return scale, translation


class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim=64, num_layers=2, flip_mask=False):
        super().__init__()
        self.dim = dim
        mask = torch.zeros(dim)
        mask[::2] = 1
        if flip_mask:
            mask = 1 - mask
        self.register_buffer("mask", mask.bool())

        cond_dim = int(self.mask.sum().item())
        trans_dim = dim - cond_dim
        self.st_net = ScaleTranslateNet(cond_dim, trans_dim, hidden_dim, num_layers)

    def forward(self, x):
        x1 = x[:, self.mask]
        x2 = x[:, ~self.mask]
        scale, translation = self.st_net(x1)
        y2 = x2 * torch.exp(scale) + translation
        y = x.clone()
        y[:, ~self.mask] = y2
        return y, scale.sum(dim=1)

    def inverse(self, y):
        y1 = y[:, self.mask]
        y2 = y[:, ~self.mask]
        scale, translation = self.st_net(y1)
        x2 = (y2 - translation) * torch.exp(-scale)
        x = y.clone()
        x[:, ~self.mask] = x2
        return x, -scale.sum(dim=1)


class RealNVP(nn.Module):
    def __init__(self, dim, num_flows, hidden_dim, num_layers):
        super().__init__()
        self.dim = dim
        self.flows = nn.ModuleList([
            CouplingLayer(dim, hidden_dim, num_layers, flip_mask=(i % 2 == 1))
            for i in range(num_flows)
        ])

    def inverse(self, z):
        x = z
        log_det = torch.zeros(z.size(0), device=z.device)
        for flow in reversed(self.flows):
            x, ld = flow.inverse(x)
            log_det += ld
        return x, log_det


class TorchMollifier:
    def __call__(self, u, x):
        u = u * torch.sin(np.pi * x[..., 0]) * torch.sin(np.pi * x[..., 1])
        return u.unsqueeze(-1)


# =============================================================================
# Data loading
# =============================================================================

def load_npy_data(path, dtype=torch.float32):
    """Load data from npy folder structure."""
    path = Path(path)

    a = np.load(path / 'coeff.npy')  # (29, 29, N)
    u = np.load(path / 'sol.npy')  # (29, 29, N)
    X = np.load(path / 'X.npy')  # (29, 29)
    Y = np.load(path / 'Y.npy')  # (29, 29)

    # Reshape: (29, 29, N) -> (N, 29*29)
    ndata = a.shape[-1]
    a = torch.from_numpy(a.reshape(-1, ndata).T).to(dtype)  # (N, 841)
    u = torch.from_numpy(u.reshape(-1, ndata).T).to(dtype)  # (N, 841)

    # Grid
    gridx = torch.from_numpy(np.stack([X.ravel(), Y.ravel()], axis=-1)).to(dtype)  # (841, 2)

    a = a.reshape(ndata, -1, 1)  # (N, 841, 1)
    u = u.reshape(ndata, -1, 1)  # (N, 841, 1)
    x = gridx.unsqueeze(0).repeat(ndata, 1, 1)  # (N, 841, 2)

    print(f"  Loaded: {ndata} samples, {a.shape[1]} points (29x29 grid)")

    return {'a': a, 'u': u, 'x': x}


# =============================================================================
# Build models (matching your architecture)
# =============================================================================

def build_models(dtype=torch.float32):
    """Build models matching your DarcyContinuous architecture."""
    BETA_SIZE = 128
    HIDDEN_SIZE = 100

    # Encoder
    conv_arch = [1, 64, 64, 64]
    fc_arch = [64 * 2 * 2, 128, 128, BETA_SIZE]
    model_enc = EncoderCNNet2dTanh(
        conv_arch=conv_arch, fc_arch=fc_arch,
        activation_conv='SiLU', activation_fc='SiLU',
        nx_size=29, ny_size=29, kernel_size=(3, 3), stride=2, dtype=dtype
    )

    # Decoders
    trunk_layers = [HIDDEN_SIZE] * 6
    branch_layers = [HIDDEN_SIZE] * 6

    model_a = MultiONetBatch(
        in_size_x=2, in_size_a=BETA_SIZE,
        trunk_layers=trunk_layers, branch_layers=branch_layers,
        activation_trunk='Tanh_Sin', activation_branch='Tanh_Sin',
        sum_layers=5, dtype=dtype
    )

    model_u = MultiONetBatch(
        in_size_x=2, in_size_a=BETA_SIZE,
        trunk_layers=trunk_layers, branch_layers=branch_layers,
        activation_trunk='Tanh_Sin', activation_branch='Tanh_Sin',
        sum_layers=5, dtype=dtype
    )

    # NF
    model_nf = RealNVP(dim=BETA_SIZE, num_flows=3, hidden_dim=64, num_layers=2)

    return {'enc': model_enc, 'u': model_u, 'a': model_a, 'nf': model_nf}


# =============================================================================
# Main debug function
# =============================================================================

def debug_step_by_step(checkpoint_path: str, data_path: str):
    device = torch.device('mps')
    dtype = torch.float32
    print(f"Using device: {device}")
    print("=" * 60)

    # =========================================================================
    # STEP 1: Load checkpoint
    # =========================================================================
    print("\n[STEP 1] Loading checkpoint...")
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"  ERROR: Checkpoint not found at {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"  Checkpoint keys: {list(ckpt.keys())}")

    # =========================================================================
    # STEP 2: Build and load models
    # =========================================================================
    print("\n[STEP 2] Building and loading models...")

    models = build_models(dtype)

    # Load state dict
    # Load state dict - your checkpoint uses 'models' key
    if 'models' in ckpt:
        state_dict = ckpt['models']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    print(f"  Actual model state dict keys (first 10): {list(state_dict.keys())[:10]}")

    # Load each model - state_dict[name] contains that model's full state dict
    # Load each model - handle potential prefix mismatches
    for name, model in models.items():
        if name in state_dict:
            saved_state = state_dict[name]

            # Check if there's a prefix mismatch
            saved_keys = list(saved_state.keys())
            model_keys = list(model.state_dict().keys())

            print(f"  {name} - saved keys (first 3): {saved_keys[:3]}")
            print(f"  {name} - model keys (first 3): {model_keys[:3]}")

            # Try to strip 'net.' prefix if present
            if saved_keys and saved_keys[0].startswith('net.') and not model_keys[0].startswith('net.'):
                saved_state = {k.replace('net.', '', 1): v for k, v in saved_state.items()}
                # Also handle nested 'net.fc_net.net.' -> 'fc_net.'
                saved_state = {k.replace('fc_net.net.', 'fc_net.'): v for k, v in saved_state.items()}
                print(f"  {name} - stripped 'net.' prefix")

            try:
                model.load_state_dict(saved_state, strict=False)
                print(f"  Loaded {name}")
            except Exception as e:
                print(f"  ERROR loading {name}: {e}")
        else:
            print(f"  WARNING: No parameters found for {name}")
        model.to(device)
        model.eval()

    # =========================================================================
    # STEP 3: Load test data
    # =========================================================================
    print("\n[STEP 3] Loading test data...")

    try:
        test_data = load_npy_data(data_path, dtype)
        print(f"  Loaded data: a={test_data['a'].shape}, u={test_data['u'].shape}")
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        print("  Trying to generate synthetic test data...")

        # Generate simple synthetic data for testing
        n_points = 29 * 29
        x_1d = torch.linspace(0, 1, 29)
        X, Y = torch.meshgrid(x_1d, x_1d, indexing='ij')
        x = torch.stack([X.flatten(), Y.flatten()], dim=-1).unsqueeze(0)

        # Simple test coefficient: a = 2.1 + sin(2*pi*x) + cos(2*pi*y)
        a = 2.1 + torch.sin(2 * np.pi * X) + torch.cos(2 * np.pi * Y)
        a = a.flatten().unsqueeze(0).unsqueeze(-1)

        # Dummy u (we'll compute from model)
        u = torch.zeros(1, n_points, 1)

        test_data = {'a': a, 'u': u, 'x': x}
        print(f"  Generated synthetic data: a={a.shape}")

    # Get first sample
    a_true = test_data['a'][0:1].to(device)
    u_true = test_data['u'][0:1].to(device)
    x = test_data['x'][0:1].to(device)

    print(f"  a_true: shape={a_true.shape}, range=[{a_true.min():.4f}, {a_true.max():.4f}]")
    print(f"  u_true: shape={u_true.shape}, range=[{u_true.min():.4f}, {u_true.max():.4f}]")

    mollifier = TorchMollifier()

    # =========================================================================
    # STEP 4: Test encoder -> decoder roundtrip
    # =========================================================================
    print("\n[STEP 4] Testing encoder -> decoder roundtrip...")
    print("  (If this fails, DGNO training didn't work)")

    with torch.no_grad():
        beta_encoded = models['enc'](a_true)
        print(f"  beta_encoded: shape={beta_encoded.shape}")
        print(f"  beta_encoded: range=[{beta_encoded.min():.4f}, {beta_encoded.max():.4f}]")
        print(f"  beta_encoded: mean={beta_encoded.mean():.4f}, std={beta_encoded.std():.4f}")

        # Decode to coefficient
        a_recon = models['a'](x, beta_encoded)
        a_true_flat = a_true.squeeze(-1)

        recon_error = (torch.norm(a_recon - a_true_flat) / torch.norm(a_true_flat)).item()
        print(f"  Coefficient reconstruction error: {recon_error:.6f}")

        # Decode to solution
        u_pred = models['u'](x, beta_encoded)
        u_pred = mollifier(u_pred, x)

        if u_true.abs().sum() > 0:  # Only compute if we have real u data
            u_error = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
            print(f"  Solution prediction error: {u_error:.6f}")

    if recon_error > 0.1:
        print("\n  *** WARNING: High reconstruction error! ***")
        print("  Your DGNO encoder-decoder is NOT working.")
        print("  Check your DGNO training.")
    else:
        print("\n  OK: Encoder-decoder looks good!")

    # =========================================================================
    # STEP 5: Test NF initialization
    # =========================================================================
    print("\n[STEP 5] Testing NF initialization...")

    nf = models['nf']

    with torch.no_grad():
        z = torch.randn(1, nf.dim, device=device)
        beta_nf, _ = nf.inverse(z)

        print(f"  z (input): range=[{z.min():.4f}, {z.max():.4f}]")
        print(f"  beta_nf: range=[{beta_nf.min():.4f}, {beta_nf.max():.4f}]")
        print(f"  beta_nf: mean={beta_nf.mean():.4f}, std={beta_nf.std():.4f}")

        # Compare distributions
        print(f"\n  Comparing NF output to encoded beta:")
        print(f"    Encoded beta: mean={beta_encoded.mean():.4f}, std={beta_encoded.std():.4f}")
        print(f"    NF beta:      mean={beta_nf.mean():.4f}, std={beta_nf.std():.4f}")

        # Decode NF sample
        a_nf = models['a'](x, beta_nf)
        print(f"  Decoded a from NF: range=[{a_nf.min():.4f}, {a_nf.max():.4f}]")
        print(f"  True a:            range=[{a_true.min():.4f}, {a_true.max():.4f}]")

        # =========================================================================
        # Setup observations for inversion
        # =========================================================================
        n_obs = 100
        np.random.seed(42)
        n_points = x.shape[1]
        obs_idx = np.random.choice(n_points, min(n_obs, n_points), replace=False)

        x_obs = x[:, obs_idx, :]
        u_obs = u_true[:, obs_idx, :]

        print(f"\n  Setup: {len(obs_idx)} observation points")
        print(f"  u_obs range: [{u_obs.min():.4f}, {u_obs.max():.4f}]")

    # =========================================================================
    # STEP 6: Running inversion optimization (500 steps)
    # =========================================================================
    print("\n[STEP 6] Running inversion optimization (500 steps)...")

    torch.manual_seed(42)

    beta_init = torch.randn(1, 128, device=device) * 0.11
    beta = nn.Parameter(beta_init.clone(), requires_grad=True)

    optimizer = torch.optim.Adam([beta], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)

    with torch.no_grad():
        a_pred = models['a'](x, beta)
        err_init = (torch.norm(a_pred - a_true_flat) / torch.norm(a_true_flat)).item()
    print(f"  Initial coefficient error: {err_init:.6f}")

    for i in range(500):
        optimizer.zero_grad()
        u_pred_obs = models['u'](x_obs, beta)
        u_pred_obs = mollifier(u_pred_obs, x_obs)
        loss_data = torch.norm(u_pred_obs - u_obs, 2, 1) / torch.norm(u_obs, 2, 1)
        loss = 50.0 * loss_data.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (i + 1) % 100 == 0:
            with torch.no_grad():
                a_pred = models['a'](x, beta)
                err = (torch.norm(a_pred - a_true_flat) / torch.norm(a_true_flat)).item()
            print(f"    Step {i + 1}: loss={loss.item():.4f}, coef_err={err:.4f}")

    with torch.no_grad():
        a_pred_final = models['a'](x, beta)
        err_final = (torch.norm(a_pred_final - a_true_flat) / torch.norm(a_true_flat)).item()

    # =========================================================================
    # STEP 7: Testing inversion from ENCODED beta (bypassing NF)
    # =========================================================================
    print("\n[STEP 7] Testing inversion from ENCODED beta (500 steps)...")

    torch.manual_seed(43)

    with torch.no_grad():
        beta_true = models['enc'](a_true).clone().detach()

    beta = nn.Parameter(beta_true + 0.01 * torch.randn_like(beta_true), requires_grad=True)

    optimizer = torch.optim.Adam([beta], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)

    with torch.no_grad():
        a_pred = models['a'](x, beta)
        err_init_7 = (torch.norm(a_pred - a_true_flat) / torch.norm(a_true_flat)).item()
    print(f"  Initial coefficient error: {err_init_7:.6f}")

    for i in range(500):
        optimizer.zero_grad()
        u_pred_obs = models['u'](x_obs, beta)
        u_pred_obs = mollifier(u_pred_obs, x_obs)
        loss_data = torch.norm(u_pred_obs - u_obs, 2, 1) / torch.norm(u_obs, 2, 1)
        loss = 50.0 * loss_data.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (i + 1) % 100 == 0:
            with torch.no_grad():
                a_pred = models['a'](x, beta)
                err = (torch.norm(a_pred - a_true_flat) / torch.norm(a_true_flat)).item()
            print(f"    Step {i + 1}: loss={loss.item():.4f}, coef_err={err:.4f}")

    with torch.no_grad():
        a_pred_final_7 = models['a'](x, beta)
        err_final_good_init = (torch.norm(a_pred_final_7 - a_true_flat) / torch.norm(a_true_flat)).item()
    print(f"  Final coefficient error (good init): {err_final_good_init:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--data', default='data/darcy_continuous/smh_test_in/',
                        help='Path to test data folder')
    args = parser.parse_args()

    debug_step_by_step(args.checkpoint, args.data)