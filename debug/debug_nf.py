import torch

ckpt = torch.load('runs/2026-01-12_19-32-43_darcy_continuous/foundation/weights/best.pt', weights_only=False, map_location='mps')



# Check what the NF does to encoded latents
# First, let's get some actual encoded latents
import numpy as np

a = np.load('data/darcy_continuous/smh_train/coeff.npy')
a = torch.from_numpy(a.reshape(-1, 1000).T).float().reshape(1000, -1, 1)

# Load encoder
from src.components.encoder import EncoderCNNet2dTanh

enc = EncoderCNNet2dTanh(
    conv_arch=[1, 64, 64, 64],
    fc_arch=[64 * 2 * 2, 128, 128, 128],
    activation_conv='SiLU', activation_fc='SiLU',
    nx_size=29, ny_size=29, kernel_size=(3, 3), stride=2
)
enc.load_state_dict(ckpt['models']['enc'])
enc.eval()

# Load NF
from src.components.nf import RealNVP

nf = RealNVP(dim=128, num_flows=3, hidden_dim=64, num_layers=2)
nf.load_state_dict(ckpt['models']['nf'])
nf.eval()

with torch.no_grad():
    # Get encoded latents
    beta_encoded = enc(a[:10])
    print(f"Encoded β: mean={beta_encoded.mean():.4f}, std={beta_encoded.std():.4f}")

    # Forward through NF (β → z)
    z_from_beta, _ = nf.forward(beta_encoded)
    print(f"NF forward (β→z): mean={z_from_beta.mean():.4f}, std={z_from_beta.std():.4f}")

    # Sample standard normal and inverse (z → β)
    z_sample = torch.randn(10, 128)
    beta_from_z, _ = nf.inverse(z_sample)
    print(f"NF inverse (z→β): mean={beta_from_z.mean():.4f}, std={beta_from_z.std():.4f}")

    # Round-trip test: β → z → β
    beta_roundtrip, _ = nf.inverse(z_from_beta)
    print(f"Round-trip β: mean={beta_roundtrip.mean():.4f}, std={beta_roundtrip.std():.4f}")
    roundtrip_error = (beta_roundtrip - beta_encoded).abs().mean()
    print(f"Round-trip error: {roundtrip_error:.6f}")

with torch.no_grad():
    # Log prob of actual encoded betas
    log_prob_encoded = nf.log_prob(beta_encoded)
    print(f"Log prob of encoded β: {log_prob_encoded.mean():.2f}")

    # Log prob of random samples with same std as encoded
    beta_random_tight = torch.randn(10, 128) * 0.11
    log_prob_tight = nf.log_prob(beta_random_tight)
    print(f"Log prob of random tight β: {log_prob_tight.mean():.2f}")

    # Log prob of standard normal samples
    beta_random_wide = torch.randn(10, 128)
    log_prob_wide = nf.log_prob(beta_random_wide)
    print(f"Log prob of random wide β: {log_prob_wide.mean():.2f}")