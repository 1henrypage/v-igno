"""
NF ablation debug script.
Trains NF for a few epochs and monitors key metrics.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# === CONFIG (modify these for ablations) ===
ABLATION = {
    'weight_decay': 0.0000,      # Try: 0.0 vs 0.0001
    'tanh_scale': 3.0,        # Try: 3.0, 5.0, 8.0
    'grad_clip': 5.0,         # Try: 1.0, 5.0, 10.0, None
    'lr': 0.001,
    'epochs': 2000,
    'print_every': 200,
}

print(f"=== ABLATION CONFIG ===")
for k, v in ABLATION.items():
    print(f"  {k}: {v}")
print()

# === Load encoder and data ===
from src.components.encoder import EncoderCNNet2dTanh
from src.utils.npy_loader import NpyFile


ckpt = torch.load('/Users/henry/school/v-igno/runs/2026-01-12_19-32-43_darcy_continuous/foundation/weights/best_dgno.pt', weights_only=False, map_location=torch.device("mps"))  # UPDATE THIS PATH

enc = EncoderCNNet2dTanh(
    conv_arch=[1, 64, 64, 64],
    fc_arch=[64 * 2 * 2, 128, 128, 128],
    activation_conv='SiLU', activation_fc='SiLU',
    nx_size=29, ny_size=29, kernel_size=(3, 3), stride=2
)
enc.load_state_dict(ckpt['models']['enc'])
enc.eval()

# Load data correctly
path = Path('data/darcy_continuous/smh_train/')
data = NpyFile(path=path, mode='r')
a = torch.from_numpy(np.array(data["coeff"]).T).float()
a = a.reshape(a.shape[0], -1, 1)

with torch.no_grad():
    latents = enc(a)
    
print(f"Latents: mean={latents.mean():.4f}, std={latents.std():.4f}")
print(f"  per-dim std: min={latents.std(0).min():.4f}, max={latents.std(0).max():.4f}")
print()

# === Define NF with configurable tanh scale ===
class FCNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RealNVPFlow(nn.Module):
    def __init__(self, dim, hidden_dim=64, num_layers=2, tanh_scale=3.0):
        super().__init__()
        self.dim = dim
        self.tanh_scale = tanh_scale

        self.t1 = FCNN(dim // 2, dim // 2, hidden_dim, num_layers)
        self.s1 = FCNN(dim // 2, dim // 2, hidden_dim, num_layers)
        self.t2 = FCNN(dim // 2, dim // 2, hidden_dim, num_layers)
        self.s2 = FCNN(dim // 2, dim // 2, hidden_dim, num_layers)

        nn.init.zeros_(self.s1.network[-1].weight)
        nn.init.zeros_(self.s1.network[-1].bias)
        nn.init.zeros_(self.s2.network[-1].weight)
        nn.init.zeros_(self.s2.network[-1].bias)

        self.log_scale_base1 = nn.Parameter(torch.zeros(dim // 2))
        self.log_scale_base2 = nn.Parameter(torch.zeros(dim // 2))

    def forward(self, x):
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]

        t1 = self.t1(lower)
        s1 = self.log_scale_base1 + torch.tanh(self.s1(lower)) * self.tanh_scale
        upper = t1 + upper * torch.exp(s1)

        t2 = self.t2(upper)
        s2 = self.log_scale_base2 + torch.tanh(self.s2(upper)) * self.tanh_scale
        lower = t2 + lower * torch.exp(s2)

        return torch.cat([lower, upper], dim=1), s1.sum(1) + s2.sum(1)

    def inverse(self, z):
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]

        t2 = self.t2(upper)
        s2 = self.log_scale_base2 + torch.tanh(self.s2(upper)) * self.tanh_scale
        lower = (lower - t2) * torch.exp(-s2)

        t1 = self.t1(lower)
        s1 = self.log_scale_base1 + torch.tanh(self.s1(lower)) * self.tanh_scale
        upper = (upper - t1) * torch.exp(-s1)

        return torch.cat([lower, upper], dim=1), -s1.sum(1) - s2.sum(1)


class RealNVP(nn.Module):
    def __init__(self, dim, num_flows=3, hidden_dim=64, num_layers=2, tanh_scale=3.0):
        super().__init__()
        self.dim = dim
        self.flows = nn.ModuleList([
            RealNVPFlow(dim, hidden_dim, num_layers, tanh_scale) 
            for _ in range(num_flows)
        ])
        self.register_buffer("log_2pi", torch.log(torch.tensor(2 * torch.pi)))

    def forward(self, x):
        log_det = torch.zeros(x.size(0), device=x.device)
        for flow in self.flows:
            x, ld = flow(x)
            log_det += ld
        return x, log_det

    def inverse(self, z):
        log_det = torch.zeros(z.size(0), device=z.device)
        for flow in reversed(self.flows):
            z, ld = flow.inverse(z)
            log_det += ld
        return z, log_det

    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + self.log_2pi).sum(dim=1)
        return log_pz + log_det

    def loss(self, x):
        return -self.log_prob(x).mean()


# === Training ===
nf = RealNVP(dim=128, num_flows=3, hidden_dim=64, num_layers=2, 
             tanh_scale=ABLATION['tanh_scale'])

optimizer = torch.optim.Adam(nf.parameters(), lr=ABLATION['lr'], 
                              weight_decay=ABLATION['weight_decay'])

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(latents),
    batch_size=50, shuffle=True
)

print("=== TRAINING ===")
for epoch in range(ABLATION['epochs']):
    nf.train()
    total_loss = 0
    
    for (batch,) in train_loader:
        loss = nf.loss(batch)
        
        optimizer.zero_grad()
        loss.backward()
        
        if ABLATION['grad_clip'] is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(nf.parameters(), ABLATION['grad_clip'])
        else:
            grad_norm = sum(p.grad.norm()**2 for p in nf.parameters() if p.grad is not None)**0.5
            
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % ABLATION['print_every'] == 0:
        nf.eval()
        with torch.no_grad():
            # Forward on real data
            z_out, _ = nf.forward(latents[:100])
            # Inverse from N(0,1)
            z_sample = torch.randn(100, 128)
            beta_out, _ = nf.inverse(z_sample)
            
            # Check scale base values
            base1_vals = [nf.flows[i].log_scale_base1.mean().item() for i in range(3)]
            
        print(f"Epoch {epoch+1:5d} | NLL: {total_loss/len(train_loader):8.2f} | "
              f"z_std: {z_out.std():.3f} | inv_std: {beta_out.std():.4f} | "
              f"base1: {base1_vals}")

# === Final evaluation ===
print("\n=== FINAL EVALUATION ===")
nf.eval()
with torch.no_grad():
    z_out, _ = nf.forward(latents[:100])
    z_sample = torch.randn(100, 128)
    beta_out, _ = nf.inverse(z_sample)
    
print(f"Target:  mean={latents.mean():.4f}, std={latents.std():.4f}")
print(f"Forward: mean={z_out.mean():.4f}, std={z_out.std():.4f} (should be ~0, ~1)")
print(f"Inverse: mean={beta_out.mean():.4f}, std={beta_out.std():.4f} (should match target)")
