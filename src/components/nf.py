
import torch
import torch.nn as nn
from typing import Tuple
import matplotlib.pyplot as plt

from src.utils.misc_utils import get_default_device

# ============================================================
# Scale + Translate Network
# ============================================================

class ScaleTranslateNet(nn.Module):
    """
    Predicts scale and translation for the transformed variables,
    conditioned on the masked variables.
    """

    def __init__(
            self,
            cond_dim: int,
            out_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2
    ):
        """
        :param cond_dim: dimension of the latent (beta) variable.
        :param out_dim: dimension of the prior (z) variable.
        :param hidden_dim: dimension of the hidden layers in the networks.
        :param num_layers: number of layers in the networks - 1
        # TODO pay attention here as well1, we don't know if the paper references total layers or only hidden layers themselves. ^^^^^

        """

        super().__init__()

        layers = [nn.Linear(cond_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]

        self.net = nn.Sequential(*layers)

        self.scale_layer = nn.Linear(hidden_dim, out_dim)
        self.translate_layer = nn.Linear(hidden_dim, out_dim)

        # VERY IMPORTANT: start as identity transform
        nn.init.zeros_(self.scale_layer.weight)
        nn.init.zeros_(self.scale_layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        # TODO idk if this needs to align with their implementation.
        scale = torch.tanh(self.scale_layer(h)) * 2.0
        translation = self.translate_layer(h)
        return scale, translation


# ============================================================
# Coupling Layer
# ============================================================

class CouplingLayer(nn.Module):
    """
    RealNVP affine coupling layer
    """

    def __init__(
            self,
            dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            flip_mask: bool = False
    ):
        super().__init__()
        self.dim = dim

        mask = torch.zeros(dim)
        mask[::2] = 1
        if flip_mask:
            mask = 1 - mask

        self.register_buffer("mask", mask.bool())

        cond_dim = self.mask.sum().item()
        trans_dim = (~self.mask).sum().item()

        self.st_net = ScaleTranslateNet(
            cond_dim=cond_dim,
            out_dim=trans_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = x[:, self.mask]
        x2 = x[:, ~self.mask]

        scale, translation = self.st_net(x1)
        y2 = x2 * torch.exp(scale) + translation

        y = x.clone()
        y[:, ~self.mask] = y2

        log_det = scale.sum(dim=1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y1 = y[:, self.mask]
        y2 = y[:, ~self.mask]

        scale, translation = self.st_net(y1)
        x2 = (y2 - translation) * torch.exp(-scale)

        x = y.clone()
        x[:, ~self.mask] = x2

        log_det = -scale.sum(dim=1)
        return x, log_det


# ============================================================
# RealNVP Model
# ============================================================

class RealNVP(nn.Module):
    """
    RealNVP normalizing flow
    """

    def __init__(
            self,
            latent_dim: int,
            num_flows: int = 6,
            hidden_dim: int = 128,
            num_layers: int = 3
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.flows = nn.ModuleList([
            CouplingLayer(
                dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                flip_mask=(i % 2 == 1)
            )
            for i in range(num_flows)
        ])

        self.register_buffer("log_2pi", torch.log(torch.tensor(2 * torch.pi)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        log_det_total = torch.zeros(x.size(0), device=x.device)

        for flow in self.flows:
            z, log_det = flow(z)
            log_det_total += log_det

        return z, log_det_total

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z
        log_det_total = torch.zeros(z.size(0), device=z.device)

        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x)
            log_det_total += log_det

        return x, log_det_total

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-likelihood for outputs: direction \beta -> z

        """
        z, log_det = self.forward(x) # output z candidate, logarithm of the determinant
        log_pz = -0.5 * (z ** 2 + self.log_2pi).sum(dim=1)
        return log_pz + log_det

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood loss

        """
        return -self.log_prob(x).mean()

    def sample(self, num_samples: int, device=None):
        if device is None:
            device = get_default_device()
        z = torch.randn(num_samples, self.latent_dim, device=device)
        x, _ = self.inverse(z)
        return x



# ====================
# TOY PROBLEM TEST
# ====================

def create_2d_mixture_data(n_samples=1000):
    """Create a 2D mixture of Gaussians as toy data"""
    n_per_component = n_samples // 3

    # Component 1: centered at (-2, -2)
    data1 = torch.randn(n_per_component, 2) * 0.5 + torch.tensor([-2.0, -2.0])

    # Component 2: centered at (2, 2)
    data2 = torch.randn(n_per_component, 2) * 0.5 + torch.tensor([2.0, 2.0])

    # Component 3: centered at (2, -2)
    data3 = torch.randn(n_samples - 2 * n_per_component, 2) * 0.5 + torch.tensor([2.0, -2.0])

    data = torch.cat([data1, data2, data3], dim=0)
    data = data[torch.randperm(data.size(0))]

    return data


def plot_results(model, data, epoch, device):
    """Visualize the learned distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original data
    axes[0, 0].scatter(data[:, 0].cpu(), data[:, 1].cpu(), alpha=0.5, s=1)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_xlim(-4, 4)
    axes[0, 0].set_ylim(-4, 4)

    # Samples from model
    with torch.no_grad():
        samples = model.sample(1000, device=device)
    axes[0, 1].scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.5, s=1, color='red')
    axes[0, 1].set_title(f'Generated Samples (Epoch {epoch})')
    axes[0, 1].set_xlim(-4, 4)
    axes[0, 1].set_ylim(-4, 4)

    # Latent space (data transformed to z)
    with torch.no_grad():
        z, _ = model.forward(data[:1000].to(device))
    axes[1, 0].scatter(z[:, 0].cpu(), z[:, 1].cpu(), alpha=0.5, s=1, color='green')
    axes[1, 0].set_title('Data in Latent Space (should be Gaussian)')
    axes[1, 0].set_xlim(-4, 4)
    axes[1, 0].set_ylim(-4, 4)

    # Log probability heatmap
    x = torch.linspace(-4, 4, 100)
    y = torch.linspace(-4, 4, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)

    with torch.no_grad():
        log_probs = model.log_prob(points).cpu().reshape(100, 100)

    im = axes[1, 1].imshow(log_probs.T, extent=[-4, 4, -4, 4], origin='lower', cmap='viridis')
    axes[1, 1].set_title('Log Probability Heatmap')
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Set device
    device = get_default_device()
    print(f"Using device: {device}")

    # Create toy data
    print("Creating toy dataset...")
    data = create_2d_mixture_data(n_samples=5000)
    train_data = data.to(device)

    # Initialize model
    print("Initializing RealNVP model...")
    model = RealNVP(
        latent_dim=2,
        num_flows=6,
        hidden_dim=128,
        num_layers=3
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    # Training loop
    print("Starting training...")
    batch_size = 256
    num_epochs = 2000

    losses = []

    for epoch in range(num_epochs):
        model.train()

        # Mini-batch training
        indices = torch.randperm(train_data.size(0))
        epoch_loss = 0
        num_batches = 0

        for i in range(0, train_data.size(0), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = train_data[batch_indices]

            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Visualize progress
        if (epoch + 1) % 500 == 0 or epoch == 0:
            model.eval()
            fig = plot_results(model, data, epoch + 1, device)
            plt.savefig(f'realnvp_epoch_{epoch + 1}.png', dpi=150, bbox_inches='tight')
            plt.close()

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Final visualization
    model.eval()
    fig = plot_results(model, data, num_epochs, device)
    plt.savefig('realnvp_final.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nTraining complete!")
    print(f"Final loss: {losses[-1]:.4f}")

    # Test forward and inverse consistency
    print("\nTesting forward-inverse consistency...")
    with torch.no_grad():
        test_samples = data[:100].to(device)
        z, log_det_fwd = model.forward(test_samples)
        reconstructed, log_det_inv = model.inverse(z)

        reconstruction_error = (test_samples - reconstructed).abs().mean()
        log_det_error = (log_det_fwd + log_det_inv).abs().mean()

        print(f"Reconstruction error: {reconstruction_error:.6f}")
        print(f"Log-det consistency: {log_det_error:.6f}")

    print("\n" + "=" * 50)
    print("Test completed successfully!")
    print("=" * 50)






