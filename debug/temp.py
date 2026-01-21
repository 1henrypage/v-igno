import torch
import numpy as np
from pathlib import Path
from src.utils.npy_loader import NpyFile
from src.components.encoder import EncoderCNNet2dTanh
from src.components.nf import RealNVP

ckpt = torch.load('/Users/henry/school/v-igno/runs/2026-01-16_21-47-39_10_just_nf/foundation/weights/best.pt', weights_only=False, map_location='mps')

enc = EncoderCNNet2dTanh(
    conv_arch=[1, 64, 64, 64],
    fc_arch=[64 * 2 * 2, 128, 128, 128],
    activation_conv='SiLU', activation_fc='SiLU',
    nx_size=29, ny_size=29, kernel_size=(3, 3), stride=2
)
enc.load_state_dict(ckpt['models']['enc'])
enc.eval()

nf = RealNVP(dim=128, num_flows=3, hidden_dim=64, num_layers=2)
nf.load_state_dict(ckpt['models']['nf'])
nf.eval()

path = Path('data/darcy_continuous/smh_train/')
data = NpyFile(path=path, mode='r')
a = torch.from_numpy(np.array(data["coeff"]).T).float()
a = a.reshape(a.shape[0], -1, 1)

with torch.no_grad():
    beta = enc(a[:100])
    
    x = beta
    for i, flow in enumerate(nf.flows):
        lower = x[:, :64]
        s1_raw = flow.s1(lower)
        
        print(f"Flow {i}: s1_raw min={s1_raw.min():.2f}, max={s1_raw.max():.2f}, tanh min={torch.tanh(s1_raw).min():.3f}, max={torch.tanh(s1_raw).max():.3f}")
        
        # Advance
        t1 = flow.t1(lower)
        upper = x[:, 64:]
        s1 = flow.log_scale_base1 + torch.tanh(s1_raw) * 8.0
        upper = t1 + upper * torch.exp(s1)
        t2 = flow.t2(upper)
        s2 = flow.log_scale_base2 + torch.tanh(flow.s2(upper)) * 8.0
        lower = t2 + lower * torch.exp(s2)
        x = torch.cat([lower, upper], dim=1)
