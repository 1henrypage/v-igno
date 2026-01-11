"""
Evaluation module for inverse problems.

Components:
- IGNOInverter: Gradient-based inversion (IGNO style)
- EncoderInverter: Direct encoding (placeholder for your method)
- Metrics: RMSE, relative L2, cross-correlation
"""
from .igno import IGNOInverter, EncoderInverter
from .metrics import (
    rmse,
    relative_l2,
    cross_correlation,
    compute_all_metrics,
    aggregate_metrics,
)

__all__ = [
    'IGNOInverter',
    'EncoderInverter',
    'rmse',
    'relative_l2',
    'cross_correlation',
    'compute_all_metrics',
    'aggregate_metrics',
]
