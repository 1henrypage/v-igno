import torch
import numpy as np

ckpt = torch.load('/Users/henry/school/v-igno/runs/2026-01-15_17-20-33_20000/foundation/weights/best.pt', weights_only=False, map_location='mps')

# Load training data to get real encoded betas
a = np.load('data/darcy_continuous/smh_train/coeff.npy')
a = torch.from_numpy(a.reshape(-1, 1000).T).float().reshape(1000, -1, 1)

from src.components.encoder import EncoderCNNet2dTanh
enc = EncoderCNNet2dTanh(
    conv_arch=[1, 64, 64, 64],
    fc_arch=[64 * 2 * 2, 128, 128, 128],
    activation_conv='SiLU', activation_fc='SiLU',
    nx_size=29, ny_size=29, kernel_size=(3, 3), stride=2
)
enc.load_state_dict(ckpt['models']['enc'])
enc.eval()

from src.components.nf import RealNVP
nf = RealNVP(dim=128, num_flows=3, hidden_dim=64, num_layers=2)
nf.load_state_dict(ckpt['models']['nf'])
nf.eval()

with torch.no_grad():
    # Get REAL encoded betas (what NF was trained on)
    beta_real = enc(a[:100])
    print(f"Real encoded beta: mean={beta_real.mean():.4f}, std={beta_real.std():.4f}")
    print(f"  per-dim std: min={beta_real.std(0).min():.4f}, max={beta_real.std(0).max():.4f}")
    
    # Forward on real betas (this is training direction)
    z_from_real, log_det = nf.forward(beta_real)
    print(f"\nNF forward(real beta) -> z:")
    print(f"  mean={z_from_real.mean():.4f}, std={z_from_real.std():.4f}")
    print(f"  min={z_from_real.min():.4f}, max={z_from_real.max():.4f}")
    
    # What's the log_prob?
    log_prob = nf.log_prob(beta_real)
    print(f"  log_prob: mean={log_prob.mean():.2f}")
    
    # Inverse from N(0,1)
    z_sample = torch.randn(100, 128)
    beta_from_z, _ = nf.inverse(z_sample)
    print(f"\nNF inverse(N(0,1)) -> beta:")
    print(f"  mean={beta_from_z.mean():.4f}, std={beta_from_z.std():.4f}")
    
    # Inverse from the ACTUAL z distribution (round-trip)
    beta_roundtrip, _ = nf.inverse(z_from_real)
    print(f"\nRound-trip (beta -> z -> beta):")
    print(f"  mean={beta_roundtrip.mean():.4f}, std={beta_roundtrip.std():.4f}")
    print(f"  reconstruction error: {(beta_roundtrip - beta_real).abs().mean():.6f}")
