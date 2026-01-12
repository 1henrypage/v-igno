
import torch
import pytest

from src.components.nf import (
    ScaleTranslateNet,
    CouplingLayer,
    RealNVP,
)


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture
def small_config():
    return {
        "dim": 2,
        "num_flows":2,
        "hidden_dim": 32,
        "num_layers": 2,
    }


@pytest.fixture
def toy_data():
    torch.manual_seed(0)

    n_samples = 256

    # Two-component Gaussian mixture
    means = torch.tensor([
        [-2.0, 0.0],
        [ 2.0, 0.0],
    ])

    std = 0.5

    # Sample mixture components
    component_ids = torch.randint(0, 2, (n_samples,))

    # Sample points
    noise = std * torch.randn(n_samples, 2)
    samples = means[component_ids] + noise

    return samples


# -------------------------
# ScaleTranslateNet tests
# -------------------------

def test_scale_translate_identity_init():
    torch.manual_seed(0)
    net = ScaleTranslateNet(cond_dim=2, out_dim=2)

    x = torch.randn(8, 2)
    scale, translation = net(x)

    # Scale should start at zero due to zero init
    assert torch.allclose(scale, torch.zeros_like(scale), atol=1e-6)

    # Translation should not be identically zero in general
    assert not torch.allclose(
        translation, torch.zeros_like(translation)
    )


# -------------------------
# CouplingLayer tests
# -------------------------

def test_coupling_layer_forward_inverse_consistency():
    torch.manual_seed(0)
    layer = CouplingLayer(dim=4, hidden_dim=32, num_layers=2)

    x = torch.randn(16, 4)
    y, log_det_fwd = layer(x)
    x_rec, log_det_inv = layer.inverse(y)

    recon_error = (x - x_rec).abs().max()
    logdet_error = (log_det_fwd + log_det_inv).abs().max()

    assert recon_error < 1e-5
    assert logdet_error < 1e-5


def test_coupling_layer_shape():
    layer = CouplingLayer(dim=6)
    x = torch.randn(10, 6)

    y, log_det = layer(x)

    assert y.shape == x.shape
    assert log_det.shape == (10,)


# -------------------------
# RealNVP tests
# -------------------------

def test_realnvp_forward_inverse_consistency(small_config):
    torch.manual_seed(0)
    model = RealNVP(small_config)

    x = torch.randn(32, small_config.dim)
    z, log_det_fwd = model.forward(x)
    x_rec, log_det_inv = model.inverse(z)

    recon_error = (x - x_rec).abs().mean()
    logdet_error = (log_det_fwd + log_det_inv).abs().mean()

    assert recon_error < 1e-5
    assert logdet_error < 1e-5


def test_realnvp_log_det_is_zero_at_init(small_config):
    torch.manual_seed(0)
    model = RealNVP(small_config)

    x = torch.randn(64, small_config.dim)
    _, log_det = model.forward(x)

    assert torch.allclose(log_det, torch.zeros_like(log_det), atol=1e-6)

def test_realnvp_sample_shape_and_finiteness(small_config, device):
    model = RealNVP(small_config)

    samples = model.sample(128, device=device)

    assert samples.shape == (128, small_config.dim)
    assert torch.isfinite(samples).all()


# -------------------------
# Training sanity check
# -------------------------

def test_training_step_reduces_loss(small_config, toy_data):
    torch.manual_seed(0)
    model = RealNVP(small_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = toy_data[:128]

    model.train()
    loss_before = model.loss(x).item()

    for _ in range(20):
        optimizer.zero_grad()
        loss = model.loss(x)
        loss.backward()
        optimizer.step()

    loss_after = model.loss(x).item()

    assert loss_after < loss_before
