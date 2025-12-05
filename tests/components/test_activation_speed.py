import time
import torch
import pytest
from src.utils.torch_utils import get_default_device

# Import our optimized FunActivation
from src.components.activation import FunActivation as MyFunActivation

# Import original FunActivation
from DGenNO.Networks.FunActivation import FunActivation as OldFunActivation

# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture(scope="module")
def device():
    return get_default_device()

@pytest.fixture(scope="module")
def sample_input(device):
    # Large tensor to measure speed differences
    return torch.randn(3000, 3000, device=device)

@pytest.fixture(scope="module")
def my_activations():
    return MyFunActivation()

@pytest.fixture(scope="module")
def old_activations():
    return OldFunActivation()

# -----------------------------
# Test equivalence and speed (with repeated calls)
# -----------------------------

def test_equivalence_and_speed(my_activations, old_activations, sample_input):
    act_names = [
        'Identity',
        'ReLU',
        'ELU',
        'Softplus',
        'Sigmoid',
        'Tanh',
        'SiLU',
        'Sinc',
        'Tanh_Sin',
        'SiLU_Sin',
        'SiLU_Id',
    ]
    repetitions = 10  # number of repeated forward passes

    print("\nActivation | Max Abs Diff | My Time (ms) | Yaohua Time (ms) | Speedup")
    print("-"*75)

    device = sample_input.device

    for name in act_names:
        # Get modules from both implementations
        my_act = my_activations(name).to(device)
        yao_act = old_activations(name).to(device)  # original keys are capitalized

        # Warmup
        _ = my_act(sample_input)
        _ = yao_act(sample_input)

        # Forward pass timing - my version
        start = time.time()
        for _ in range(repetitions):
            y_my = my_act(sample_input)
        t_my = (time.time() - start) * 1000 / repetitions  # average per call

        # Forward pass timing - yaohua version
        start = time.time()
        for _ in range(repetitions):
            y_yao = yao_act(sample_input)
        t_yao = (time.time() - start) * 1000 / repetitions  # average per call

        # Max absolute difference
        max_diff = (y_my - y_yao).abs().max().item()

        # Speedup
        speedup = t_yao / t_my if t_my > 0 else float('inf')

        print(f"{name:<10} | {max_diff:>12.6f} | {t_my:>12.3f} | {t_yao:>12.3f} | {speedup:>7.2f}")

        # Assert outputs are close
        assert torch.allclose(y_my, y_yao, atol=1e-6), f"{name} outputs differ between my and yaohua's implementations"
