import torch
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
CHECKPOINT_PATH = '/home/henry/school/v-igno/runs/100_dims/foundation/weights/best_dgno.pt'
LATENT_DIM = 100  # Update this to match your new architecture

ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location='cpu')

# Load training data
raw = np.load('data/darcy_continuous/smh_train/coeff.npy')  # (29, 29, 1000)
a = torch.from_numpy(raw.T).float()  # (1000, 29, 29)
a = a.reshape(a.shape[0], -1, 1)     # (1000, 841, 1)

# Load encoder - update fc_arch to match your new architecture
from src.components.encoder import EncoderCNNet2dTanh
enc = EncoderCNNet2dTanh(
    conv_arch=[1, 64, 64, 64],
    fc_arch=[64 * 2 * 2, 128, 128, LATENT_DIM],  # Updated for new latent dim
    activation_conv='SiLU', activation_fc='SiLU',
    nx_size=29, ny_size=29, kernel_size=(3, 3), stride=2
)
enc.load_state_dict(ckpt['models']['enc'])
enc.eval()

with torch.no_grad():
    beta = enc(a)
    
    print("=" * 60)
    print("ENCODER LATENT STATISTICS")
    print("=" * 60)
    
    # Overall stats
    print(f"\nOverall:")
    print(f"  Shape: {beta.shape}")
    print(f"  Mean: {beta.mean():.4f}")
    print(f"  Std:  {beta.std():.4f}")
    print(f"  Range: [{beta.min():.3f}, {beta.max():.3f}]")
    
    # Per-dimension stats
    per_dim_std = beta.std(dim=0)
    per_dim_mean = beta.mean(dim=0)
    
    print(f"\nPer-dimension std:")
    print(f"  Min:  {per_dim_std.min():.6f}")
    print(f"  Max:  {per_dim_std.max():.4f}")
    print(f"  Mean: {per_dim_std.mean():.4f}")
    
    # Dead dimension analysis
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    print(f"\nDead dimensions by threshold:")
    for thresh in thresholds:
        dead = (per_dim_std < thresh).sum().item()
        pct = 100 * dead / LATENT_DIM
        print(f"  std < {thresh}: {dead}/{LATENT_DIM} ({pct:.1f}%)")
    
    # Utilization of [-1, 1] range (tanh output)
    print(f"\nRange utilization (tanh outputs to [-1,1]):")
    print(f"  Dims with max |value| > 0.9: {(beta.abs().max(dim=0).values > 0.9).sum().item()}")
    print(f"  Dims with max |value| > 0.5: {(beta.abs().max(dim=0).values > 0.5).sum().item()}")
    print(f"  Dims with max |value| < 0.1: {(beta.abs().max(dim=0).values < 0.1).sum().item()}")
    
    # Effective dimensionality (how many dims carry meaningful variance)
    sorted_std, _ = per_dim_std.sort(descending=True)
    cumulative_var = (sorted_std ** 2).cumsum(dim=0) / (sorted_std ** 2).sum()
    dims_for_90 = (cumulative_var < 0.9).sum().item() + 1
    dims_for_95 = (cumulative_var < 0.95).sum().item() + 1
    dims_for_99 = (cumulative_var < 0.99).sum().item() + 1
    
    print(f"\nEffective dimensionality:")
    print(f"  Dims for 90% variance: {dims_for_90}")
    print(f"  Dims for 95% variance: {dims_for_95}")
    print(f"  Dims for 99% variance: {dims_for_99}")
    
    # Correlation analysis (are dims redundant?)
    corr_matrix = torch.corrcoef(beta.T)
    corr_matrix.fill_diagonal_(0)  # Ignore self-correlation
    high_corr = (corr_matrix.abs() > 0.8).sum().item() // 2  # Divide by 2 for symmetry
    
    print(f"\nRedundancy (correlation > 0.8): {high_corr} dim pairs")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Histogram of per-dim std
    axes[0, 0].hist(per_dim_std.numpy(), bins=30, edgecolor='black')
    axes[0, 0].axvline(x=0.01, color='r', linestyle='--', label='Dead threshold (0.01)')
    axes[0, 0].set_xlabel('Per-dimension Std')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Per-Dimension Std')
    axes[0, 0].legend()
    
    # 2. Sorted std (scree plot style)
    axes[0, 1].plot(sorted_std.numpy())
    axes[0, 1].axhline(y=0.01, color='r', linestyle='--', label='Dead threshold (0.01)')
    axes[0, 1].set_xlabel('Dimension (sorted by std)')
    axes[0, 1].set_ylabel('Std')
    axes[0, 1].set_title('Sorted Per-Dimension Std')
    axes[0, 1].legend()
    
    # 3. Cumulative variance
    axes[1, 0].plot(cumulative_var.numpy())
    axes[1, 0].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    axes[1, 0].axvline(x=dims_for_95, color='g', linestyle='--', label=f'{dims_for_95} dims')
    axes[1, 0].set_xlabel('Number of dimensions')
    axes[1, 0].set_ylabel('Cumulative variance fraction')
    axes[1, 0].set_title('Cumulative Variance Explained')
    axes[1, 0].legend()
    
    # 4. Sample of latent distributions (first 10 dims)
    for i in range(min(10, LATENT_DIM)):
        axes[1, 1].hist(beta[:, i].numpy(), bins=30, alpha=0.3, label=f'Dim {i}')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Distribution of First 10 Dimensions')
    
    plt.tight_layout()
    plt.savefig('encoder_latent_analysis.png', dpi=150)
    plt.show()
    
    print(f"\nPlot saved to encoder_latent_analysis.png")
