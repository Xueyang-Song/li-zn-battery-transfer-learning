"""Normalisation strategies for the battery ML feature pipeline.

Exposes a single entry-point ``build_scalers`` that returns per-chemistry and
unified StandardScalers, following the strategy that produced the published
results.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def build_per_chemistry_scalers(
    X_li: np.ndarray,
    X_zn: np.ndarray,
) -> tuple[StandardScaler, StandardScaler]:
    """Fit independent StandardScalers for Li-ion and Zn-ion features.

    Preserves rank-order correspondence within each chemistry but does **not**
    align the two feature spaces.  Useful as a baseline comparison.

    Args:
        X_li: Li-ion raw feature matrix, shape ``(n_li, d)``.
        X_zn: Zn-ion raw feature matrix, shape ``(n_zn, d)``.

    Returns:
        Pair ``(scaler_li, scaler_zn)`` fitted on their respective inputs.
    """
    scaler_li = StandardScaler().fit(X_li)
    scaler_zn = StandardScaler().fit(X_zn)
    return scaler_li, scaler_zn


def build_unified_scaler(
    X_li: np.ndarray,
    X_zn: np.ndarray,
) -> StandardScaler:
    """Fit a single StandardScaler over the combined Li+Zn feature space.

    Ensures that both chemistries occupy the same standardised region, which
    is required for the MT-GP kernel length-scales to be meaningful across
    tasks.

    Args:
        X_li: Li-ion raw feature matrix, shape ``(n_li, d)``.
        X_zn: Zn-ion raw feature matrix, shape ``(n_zn, d)``.

    Returns:
        ``StandardScaler`` fitted on ``vstack([X_li, X_zn])``.
    """
    X_combined = np.vstack([X_li, X_zn])
    return StandardScaler().fit(X_combined)
