import torch
import numpy as np

ckpt = torch.load('/Users/henry/school/v-igno/runs/2026-01-16_19-44-48_10_just_nf/foundation/weights/best.pt', weights_only=False, map_location='mps')

# Load training data
raw = np.load('data/darcy_continuous/smh_train/coeff.npy')  # (29, 29, 1000)
a = torch.from_numpy(raw.T).float()  # (1000, 29, 29)
a = a.reshape(a.shape[0], -1, 1)     # (1000, 841, 1)

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

# Load new NF
from src.components.nf import RealNVP
nf = RealNVP(dim=128, num_flows=3, hidden_dim=64, num_layers=2)
nf.load_state_dict(ckpt['models']['nf'])
nf.eval()

with torch.no_grad():
    beta_encoded = enc(a[:100])
    print(f"Encoded beta: mean={beta_encoded.mean():.4f}, std={beta_encoded.std():.4f}")
    
    z_out, _ = nf.forward(beta_encoded)
    print(f"NF forward:   mean={z_out.mean():.4f}, std={z_out.std():.4f}  (should be ~0, ~1)")
    
    z_sample = torch.randn(100, 128)
    beta_out, _ = nf.inverse(z_sample)
    print(f"NF inverse:   mean={beta_out.mean():.4f}, std={beta_out.std():.4f}  (should match encoded)")
