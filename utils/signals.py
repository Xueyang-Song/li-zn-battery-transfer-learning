"""Signal-processing utilities for battery capacity and dQ/dV curves.

Refactored from ``code/utils.py``.  All functions are pure (no global state,
no I/O) to maximise testability.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def uniform_voltage_grid(
    v_min: float,
    v_max: float,
    n_points: int = 500,
) -> np.ndarray:
    """Return a uniform 1-D voltage grid.

    Args:
        v_min: Minimum voltage (inclusive).
        v_max: Maximum voltage (inclusive).
        n_points: Number of grid points.

    Returns:
        1-D array of length ``n_points`` spanning ``[v_min, v_max]``.
    """
    return np.linspace(v_min, v_max, n_points)


def capacity_to_soh(
    capacity: np.ndarray,
    reference_capacity: float | None = None,
) -> np.ndarray:
    """Normalise a capacity trace to State-of-Health (SOH) in [0, 1].

    Args:
        capacity: 1-D array of per-cycle capacity values.
        reference_capacity: Denominator used for normalisation.  Defaults to
            ``capacity[0]`` if ``None``.

    Returns:
        SOH array of the same shape as ``capacity``.

    Raises:
        ValueError: If ``reference_capacity`` is zero or ``capacity`` is empty.
    """
    if len(capacity) == 0:
        return capacity.copy()
    ref = reference_capacity if reference_capacity is not None else float(capacity[0])
    if ref == 0.0:
        raise ValueError("Reference capacity is zero — cannot normalise.")
    return capacity / ref


def find_cycle_life(
    capacity_norm: np.ndarray,
    threshold: float = 0.8,
) -> int:
    """Find the first cycle where normalised capacity falls below *threshold*.

    Args:
        capacity_norm: 1-D normalised capacity (SOH) array.
        threshold: End-of-life capacity threshold (default 0.8 = 80 % SOH).

    Returns:
        0-indexed cycle index of first failure.  Returns ``len(capacity_norm)``
        if the threshold is never crossed.
    """
    below = np.where(capacity_norm < threshold)[0]
    return int(below[0]) if len(below) > 0 else len(capacity_norm)


def monotone_fraction(arr: np.ndarray) -> float:
    """Fraction of consecutive pairs where ``arr[i+1] <= arr[i]`` (non-increasing).

    Args:
        arr: 1-D numeric array.

    Returns:
        Value in ``[0, 1]``; 1.0 means perfectly non-increasing.
    """
    if len(arr) < 2:
        return 1.0
    diffs = np.diff(arr)
    return float(np.sum(diffs <= 0)) / len(diffs)


def safe_savgol(
    y: np.ndarray,
    window: int = 11,
    order: int = 3,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing, gracefully shrinking window if needed.

    Args:
        y: 1-D signal to smooth.
        window: Desired window length (must be odd and >= ``order + 2``).
        order: Polynomial order for the SG filter.

    Returns:
        Smoothed array of the same length as ``y``.
    """
    n = len(y)
    w = window
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w < order + 2:
        w = order + 2
        if w % 2 == 0:
            w += 1
    if w > n:
        return y.copy()
    return savgol_filter(y, window_length=w, polyorder=order)


def interpolate_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate ``(x, y)`` onto ``x_grid``, handling non-monotone inputs.

    Out-of-range values are held constant at the boundary (no NaN extrapolation).

    Args:
        x: Input x coordinates (need not be sorted).
        y: Input y values, same length as ``x``.
        x_grid: Target grid to interpolate onto.

    Returns:
        Interpolated values on ``x_grid``.

    Raises:
        ValueError: If ``x`` and ``y`` have different lengths.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    _, unique_idx = np.unique(xs, return_index=True)
    xs, ys = xs[unique_idx], ys[unique_idx]
    return np.interp(x_grid, xs, ys)


def compute_dqdv_on_grid(
    voltage: np.ndarray,
    capacity: np.ndarray,
    v_grid: np.ndarray,
    sg_window: int = 11,
    sg_order: int = 3,
) -> np.ndarray:
    """Compute smoothed dQ/dV interpolated onto a common voltage grid.

    Steps:
    1. Interpolate ``capacity`` onto ``v_grid``.
    2. Compute numerical gradient w.r.t. the grid.
    3. Apply Savitzky-Golay smoothing.

    Args:
        voltage: Raw voltage array for a single cycle.
        capacity: Raw capacity array for the same cycle.
        v_grid: Target voltage grid (monotonically increasing).
        sg_window: SG filter window length.
        sg_order: SG filter polynomial order.

    Returns:
        Smoothed dQ/dV values on ``v_grid``, same length as ``v_grid``.
    """
    q_interp = interpolate_to_grid(voltage, capacity, v_grid)
    dqdv = np.gradient(q_interp, v_grid)
    return safe_savgol(dqdv, window=sg_window, order=sg_order)
