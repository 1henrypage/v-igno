



import torch
from src.components.nf import RealNVP

ckpt = torch.load('/Users/henry/school/v-igno/runs/2026-01-15_15-33-35_darcy_continuous_just_nf_only_clipping/foundation/weights/best.pt', weights_only=False, map_location='mps')

nf = RealNVP(dim=128, num_flows=3, hidden_dim=64, num_layers=2)
nf.load_state_dict(ckpt['models']['nf'])
nf.eval()

x = torch.randn(10, 128) * 0.1

with torch.no_grad():
    z = x.clone()
    
    for i, flow in enumerate(nf.flows):
        lower, upper = z[:, :64], z[:, 64:]
        
        t1 = flow.t1(lower)
        s1 = flow.log_scale_base1 + flow.s1(lower)
        upper_new = t1 + upper * torch.exp(s1)
        
        t2 = flow.t2(upper_new)
        s2 = flow.log_scale_base2 + flow.s2(upper_new)
        lower_new = t2 + lower * torch.exp(s2)
        
        z = torch.cat([lower_new, upper_new], dim=1)
        
        print(f"Flow {i}:")
        print(f"  s1: min={s1.min():.2f}, max={s1.max():.2f}")
        print(f"  s2: min={s2.min():.2f}, max={s2.max():.2f}")
        print(f"  z: mean={z.mean():.4f}, std={z.std():.4f}, min={z.min():.2f}, max={z.max():.2f}")
        print(f"  has NaN: {torch.isnan(z).any()}, has Inf: {torch.isinf(z).any()}")
