import torch
import numpy as np

# Load checkpoint and get encoder
ckpt = torch.load('runs/2026-01-12_19-32-43_darcy_continuous/foundation/weights/best_dgno.pt', weights_only=False, map_location='mps')

from src.components.nf import RealNVP

nf = RealNVP(dim=128, num_flows=3, hidden_dim=64, num_layers=2)
nf.load_state_dict(ckpt['models']['nf'])
nf.eval()

# Check what forward does to UNIFORM samples in [-1,1] (like tanh output)
with torch.no_grad():
    beta_uniform = torch.rand(1000, 128) * 2 - 1  # Uniform in [-1, 1]
    z_out, _ = nf.forward(beta_uniform)
    print(f"Uniform [-1,1] input -> NF forward output: mean={z_out.mean():.4f}, std={z_out.std():.4f}")

    # And what inverse does to N(0,1)
    z_normal = torch.randn(1000, 128)
    beta_out, _ = nf.inverse(z_normal)
    print(f"N(0,1) input -> NF inverse output: mean={beta_out.mean():.4f}, std={beta_out.std():.4f}")