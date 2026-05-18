"""Evaluation metrics for GP regression in cycle-space.

All functions are pure and stateless — no side effects, no I/O.
"""

from __future__ import annotations

import numpy as np


def rmse_cycles(
    y_pred_log: np.ndarray,
    y_true_log: np.ndarray,
) -> float:
    """Root-mean-squared error in **cycle space** (i.e. after exp transform).

    Args:
        y_pred_log: Predicted values in log-cycle space.
        y_true_log: Ground-truth values in log-cycle space.

    Returns:
        RMSE computed after ``exp`` transforming both arrays.
    """
    return float(np.sqrt(np.mean((np.exp(y_true_log) - np.exp(y_pred_log)) ** 2)))


def nll_gaussian(
    mu: np.ndarray,
    var: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Mean negative log-likelihood under a diagonal Gaussian predictive.

    NLL = 0.5 * mean[ log(2π σ²) + (y - μ)² / σ² ]

    Args:
        mu: Predictive mean, shape ``(n,)``.
        var: Predictive variance, shape ``(n,)``; must be positive.
        y_true: Ground-truth targets, shape ``(n,)``.

    Returns:
        Scalar mean NLL (lower is better).
    """
    var_safe = np.clip(var, 1e-12, None)
    return float(np.mean(0.5 * (np.log(2 * np.pi * var_safe) + (y_true - mu) ** 2 / var_safe)))


def coverage_90(
    mu_log: np.ndarray,
    var_log: np.ndarray,
    y_true_log: np.ndarray,
) -> float:
    """Empirical 90 % prediction-interval coverage in log-cycle space.

    Interval: ``[μ − 1.645σ, μ + 1.645σ]``.

    Args:
        mu_log: Predictive mean in log-cycle space, shape ``(n,)``.
        var_log: Predictive variance in log-cycle space, shape ``(n,)``.
        y_true_log: Ground-truth log-cycle values, shape ``(n,)``.

    Returns:
        Fraction of test points inside the 90 % PI (should be ≈ 0.90 if calibrated).
    """
    std = np.sqrt(np.clip(var_log, 1e-12, None))
    lo = mu_log - 1.645 * std
    hi = mu_log + 1.645 * std
    return float(np.mean((y_true_log >= lo) & (y_true_log <= hi)))


def mean_pred_std(var_log: np.ndarray) -> float:
    """Mean predictive standard deviation (sharpness measure).

    Args:
        var_log: Predictive variance in log-cycle space, shape ``(n,)``.

    Returns:
        Mean of ``sqrt(var_log)`` — smaller is sharper.
    """
    return float(np.mean(np.sqrt(np.clip(var_log, 1e-12, None))))
