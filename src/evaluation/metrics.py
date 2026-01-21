"""
Evaluation metrics for inverse problems.

From IGNO paper Section 4.5:
- Relative RMSE (Eq. 14): For continuous coefficients
- Cross-correlation indicator I_corr (Eq. 15): For discontinuous targets

Also includes standard relative L2 using MyError.
"""
import torch
from typing import Dict, List

from src.utils.Losses import MyError


def rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Relative Root Mean Square Error (IGNO Eq. 14).

    RMSE = sqrt(sum((a_rec - a_true)^2) / sum(a_true^2))

    This is normalized by the energy of the true signal.
    """
    pred_flat = pred.flatten().float()
    true_flat = true.flatten().float()

    numerator = ((pred_flat - true_flat) ** 2).sum()
    denominator = (true_flat ** 2).sum()

    if denominator < 1e-10:
        return float('inf')

    return torch.sqrt(numerator / denominator).item()


def relative_l2(pred: torch.Tensor, true: torch.Tensor, p: int = 2) -> float:
    """
    Relative Lp error using MyError.

    rel_L2 = ||pred - true||_p / ||true||_p
    """
    error_fn = MyError(d=2, p=p, size_average=True, reduction=True)
    return error_fn.Lp_rel(pred, true).item()


def cross_correlation(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Cross-correlation indicator I_corr (IGNO Eq. 15).

    For discontinuous targets, measures morphological similarity.

    I_corr = sum(a_true_scaled^2 * a_rec_scaled^2) /
             (sqrt(sum(a_true_scaled^2)) * sqrt(sum(a_rec_scaled^2)))

    Where a_scaled denotes coefficients rescaled to [0, 1].

    I_corr ranges from 0 to 1, with values close to 1 indicating
    strong morphological agreement.
    """
    pred_flat = pred.flatten().float()
    true_flat = true.flatten().float()

    # Rescale to [0, 1]
    def rescale_01(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-10:
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    pred_scaled = rescale_01(pred_flat)
    true_scaled = rescale_01(true_flat)

    # Compute I_corr from Eq. 15
    # Numerator: sum(a_true^2 * a_rec^2)
    # Denominator: sqrt(sum(a_true^2)) * sqrt(sum(a_rec^2))
    true_sq = true_scaled ** 2
    pred_sq = pred_scaled ** 2

    numerator = (true_sq * pred_sq).sum()
    denominator = torch.sqrt(true_sq.sum()) * torch.sqrt(pred_sq.sum())

    if denominator < 1e-10:
        return 0.0

    return (numerator / denominator).item()


def compute_all_metrics(pred: torch.Tensor, true: torch.Tensor) -> Dict[str, float]:
    """
    Compute all metrics at once.

    Args:
        pred: Predicted values
        true: Ground truth values

    Returns:
        Dictionary with all metrics
    """

    #pred.shape == true.shape == [number_of_observations, 1], I checked this
    return {
        'rmse': rmse(pred, true),
        'relative_l2': relative_l2(pred, true),
        'cross_correlation': cross_correlation(pred, true),
    }


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple samples.

    Args:
        metrics_list: List of metric dicts from compute_all_metrics

    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    result = {}

    for key in keys:
        values = [m[key] for m in metrics_list]
        n = len(values)
        mean_val = sum(values) / n
        std_val = (sum((v - mean_val) ** 2 for v in values) / n) ** 0.5

        result[key] = {
            'mean': mean_val,
            'std': std_val,
            'min': min(values),
            'max': max(values),
        }

    return result