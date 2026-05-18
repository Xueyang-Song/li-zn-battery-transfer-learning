"""Canonical feature engineering for the battery ML pipeline.

The two features used in all published experiments are:

- ``exp_b``: early exponential decay rate from Q(n)/Q0 = A + (1-A)·exp(−b·n)
- ``delta_Q_var``: variance of SG-smoothed dQ/dV curves compressed with log10

Both were validated to transfer across Li-ion and Zn-ion chemistries.

Refactored from ``code/experiments.py`` and ``code/fpca_features.py``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler

from config.settings import PipelineSettings
from utils.signals import safe_savgol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# exp_b feature
# ---------------------------------------------------------------------------

def _exp_decay(n: np.ndarray, A: float, b: float) -> np.ndarray:
    """Functional form Q(n)/Q0 = A + (1-A)·exp(−b·n)."""
    return A + (1.0 - A) * np.exp(-b * n)


def fit_exp_b(
    capacity_curve: np.ndarray,
    n_cycles: int = 20,
) -> float:
    """Fit exponential decay rate *b* to the early post-activation capacity.

    Model: Q(n)/Q0 = A + (1-A)·exp(−b·n), fitted to ``n_cycles`` cycles.

    Args:
        capacity_curve: 1-D capacity array starting at the activation-end
            cycle (Q0 = capacity_curve[0]).
        n_cycles: Number of cycles from the start to use for fitting.

    Returns:
        Fitted decay rate ``b`` in ``[1e-6, 5.0]``.  Returns ``np.nan`` if
        the curve is too short or fitting fails.
    """
    if len(capacity_curve) < 3:
        logger.debug("Curve too short for exp_b fit (len=%d)", len(capacity_curve))
        return float("nan")

    n_fit = min(n_cycles, len(capacity_curve))
    q0 = float(capacity_curve[0])
    if q0 <= 0:
        return float("nan")

    n_arr = np.arange(n_fit, dtype=float)
    q_norm = capacity_curve[:n_fit] / q0

    try:
        popt, _ = curve_fit(
            _exp_decay,
            n_arr,
            q_norm,
            p0=[0.2, 0.05],
            bounds=([0.0, 1e-6], [1.1, 5.0]),
            maxfev=2000,
        )
        b_fit = float(popt[1])
    except (RuntimeError, ValueError) as exc:
        logger.debug("curve_fit failed: %s", exc)
        return float("nan")

    return b_fit


# ---------------------------------------------------------------------------
# delta_Q_var feature
# ---------------------------------------------------------------------------

def compute_delta_Q_var(
    capacity_curve: np.ndarray,
    sg_window: int = 11,
    sg_order: int = 3,
) -> float:
    """Variance of SG-smoothed first differences of a capacity curve.

    Args:
        capacity_curve: 1-D per-cycle capacity (or SOH) array.
        sg_window: Savitzky-Golay window length (odd integer).
        sg_order: Savitzky-Golay polynomial order.

    Returns:
        Variance of the smoothed differences, or ``np.nan`` if the curve is
        too short to compute differences.
    """
    if len(capacity_curve) < 2:
        return float("nan")

    smoothed = safe_savgol(capacity_curve, window=sg_window, order=sg_order)
    diffs = np.diff(smoothed)
    return float(np.var(diffs))


# ---------------------------------------------------------------------------
# Log10 transform
# ---------------------------------------------------------------------------

def log10_transform(x: float, eps: float = 1e-6) -> float:
    """Compress scale via log10(x + eps).

    Used to bridge the ~1000× gap between Li-ion and Zn-ion delta_Q_var
    magnitudes.

    Args:
        x: Raw feature value (must be ≥ 0).
        eps: Small constant to prevent log(0) (default 1e-6).

    Returns:
        ``log10(x + eps)``
    """
    return float(np.log10(x + eps))


# ---------------------------------------------------------------------------
# Unified feature matrix builder
# ---------------------------------------------------------------------------

FEAT_COLS = ["exp_b", "delta_Q_var"]


def build_feature_matrix(
    li_df: pd.DataFrame,
    zn_df: pd.DataFrame,
    settings: PipelineSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Build the normalised feature matrices used in GP experiments.

    Pipeline:
    1. Drop NaN rows and boundary-hit exp_b values from Li-ion data.
    2. Apply log10 transform to ``delta_Q_var`` for both chemistries.
    3. Fit a unified ``StandardScaler`` on the combined ``[Li, Zn]`` feature
       space.
    4. Return scaled arrays alongside log-cycle-life targets.

    Args:
        li_df: Li-ion feature DataFrame with columns ``exp_b``, ``delta_Q_var``,
            ``cycle_life``.
        zn_df: Zn-ion feature DataFrame with the same required columns.
        settings: Pipeline configuration (thresholds, eps).

    Returns:
        Tuple ``(X_li, y_li, X_zn, y_zn, scaler)`` where:

        - ``X_li``: scaled Li-ion features, shape ``(n_li, 2)``.
        - ``y_li``: log-cycle-life for Li-ion, shape ``(n_li,)``.
        - ``X_zn``: scaled Zn-ion features, shape ``(n_zn, 2)``.
        - ``y_zn``: log-cycle-life for Zn-ion, shape ``(n_zn,)``.
        - ``scaler``: fitted ``StandardScaler`` (fit on Li+Zn combined).
    """
    # --- Li-ion filtering ---------------------------------------------------
    li_clean = li_df.dropna(subset=FEAT_COLS + ["cycle_life"]).copy()
    li_clean = li_clean[
        (li_clean["cycle_life"] > settings.min_cycle_life)
        & (li_clean["exp_b"] < settings.exp_b_filter_max)
    ]
    logger.info(
        "Li-ion source pool",
        extra={"n_cells": len(li_clean), "step": "build_feature_matrix"},
    )

    # --- Zn-ion filtering ---------------------------------------------------
    zn_clean = zn_df.dropna(subset=FEAT_COLS + ["cycle_life"]).copy()
    zn_clean = zn_clean[zn_clean["cycle_life"] > 0]
    logger.info(
        "Zn-ion pool",
        extra={"n_cells": len(zn_clean), "step": "build_feature_matrix"},
    )

    # --- log10 transform on delta_Q_var -------------------------------------
    eps = settings.exp_b_filter_eps
    li_clean["log_delta_Q_var"] = li_clean["delta_Q_var"].apply(
        lambda x: log10_transform(x, eps)
    )
    zn_clean["log_delta_Q_var"] = zn_clean["delta_Q_var"].apply(
        lambda x: log10_transform(x, eps)
    )

    feat_cols_transformed = ["exp_b", "log_delta_Q_var"]

    X_li_raw = li_clean[feat_cols_transformed].values.astype(float)
    X_zn_raw = zn_clean[feat_cols_transformed].values.astype(float)

    y_li = np.log(li_clean["cycle_life"].values.astype(float))
    y_zn = np.log(zn_clean["cycle_life"].values.astype(float))

    # --- Unified scaler fitted on combined Li+Zn pool -----------------------
    X_combined = np.vstack([X_li_raw, X_zn_raw])
    scaler = StandardScaler()
    scaler.fit(X_combined)

    X_li = scaler.transform(X_li_raw)
    X_zn = scaler.transform(X_zn_raw)

    return X_li, y_li, X_zn, y_zn, scaler
