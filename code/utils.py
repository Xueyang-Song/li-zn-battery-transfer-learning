"""Shared utilities for battery ML pipeline."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Optional, Tuple


def uniform_voltage_grid(v_min: float, v_max: float, n_points: int = 500) -> np.ndarray:
    """Return uniform voltage grid from v_min to v_max with n_points."""
    return np.linspace(v_min, v_max, n_points)


def capacity_to_soh(
    capacity: np.ndarray,
    reference_capacity: Optional[float] = None
) -> np.ndarray:
    """
    Convert capacity to SOH (State of Health) as a fraction.

    Parameters
    ----------
    capacity : array of capacity values (one per cycle)
    reference_capacity : reference value; defaults to first cycle value if None

    Returns
    -------
    soh : normalized capacity array (values in ~[0, 1])
    """
    if len(capacity) == 0:
        return capacity.copy()
    ref = reference_capacity if reference_capacity is not None else float(capacity[0])
    if ref == 0.0:
        raise ValueError("Reference capacity is zero — cannot normalize.")
    return capacity / ref


def find_cycle_life(capacity_norm: np.ndarray, threshold: float = 0.8) -> int:
    """
    Find the first cycle where normalized capacity drops below threshold.

    Parameters
    ----------
    capacity_norm : normalized capacity array (SOH)
    threshold : EOL threshold (default 0.8 = 80 % SOH)

    Returns
    -------
    cycle index (0-indexed).  Returns len(capacity_norm) if threshold never reached.
    """
    below = np.where(capacity_norm < threshold)[0]
    return int(below[0]) if len(below) > 0 else len(capacity_norm)


def monotone_fraction(arr: np.ndarray) -> float:
    """
    Return the fraction of consecutive pairs (i, i+1) where arr[i+1] <= arr[i].
    Useful as a quick 'how monotonically decreasing is this?' score in [0, 1].
    """
    if len(arr) < 2:
        return 1.0
    diffs = np.diff(arr)
    return float(np.sum(diffs <= 0)) / len(diffs)


def safe_savgol(
    y: np.ndarray,
    window: int = 11,
    order: int = 3
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing, gracefully shrinking window if too short.

    Parameters
    ----------
    y      : 1-D signal
    window : desired window length (must be odd, >= order+2)
    order  : polynomial order

    Returns
    -------
    smoothed array of same length
    """
    n = len(y)
    w = window
    # window must be odd and <= n
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
    x_grid: np.ndarray
) -> np.ndarray:
    """
    Monotone-aware 1-D interpolation of (x, y) onto x_grid.
    Handles non-monotone x by sorting first.  Out-of-range values are
    extrapolated as boundary values (no NaNs).
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    # sort by x
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    # remove duplicate x
    _, unique_idx = np.unique(xs, return_index=True)
    xs, ys = xs[unique_idx], ys[unique_idx]
    return np.interp(x_grid, xs, ys)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import traceback

    passed = 0
    failed = 0

    def _assert(cond, name, detail=''):
        global passed, failed
        if cond:
            print(f'  PASS  {name}')
            passed += 1
        else:
            print(f'  FAIL  {name}  {detail}')
            failed += 1

    print('=== utils.py unit tests ===')

    # uniform_voltage_grid
    g = uniform_voltage_grid(2.0, 4.2, 100)
    _assert(len(g) == 100, 'grid length')
    _assert(abs(g[0] - 2.0) < 1e-9, 'grid start')
    _assert(abs(g[-1] - 4.2) < 1e-9, 'grid end')

    # capacity_to_soh
    cap = np.array([3.0, 2.7, 2.4, 2.1])
    soh = capacity_to_soh(cap)
    _assert(abs(soh[0] - 1.0) < 1e-9, 'soh first=1')
    _assert(abs(soh[-1] - 0.7) < 1e-9, 'soh last=0.7')

    soh2 = capacity_to_soh(cap, reference_capacity=3.0)
    _assert(np.allclose(soh, soh2), 'soh explicit ref')

    try:
        capacity_to_soh(np.array([1.0]), reference_capacity=0.0)
        _assert(False, 'zero-ref raises')
    except ValueError:
        _assert(True, 'zero-ref raises')

    # find_cycle_life
    norm = np.array([1.0, 0.95, 0.88, 0.79, 0.70])
    cl = find_cycle_life(norm, threshold=0.8)
    _assert(cl == 3, f'cycle life=3 (got {cl})')

    cl2 = find_cycle_life(np.ones(10), threshold=0.8)
    _assert(cl2 == 10, 'cycle life never reached')

    # monotone_fraction
    mf = monotone_fraction(np.array([1.0, 0.9, 0.8, 0.7]))
    _assert(abs(mf - 1.0) < 1e-9, 'fully monotone decreasing')

    mf2 = monotone_fraction(np.array([1.0, 1.1, 0.9, 0.8]))
    _assert(abs(mf2 - 2 / 3) < 1e-6, f'partially monotone (got {mf2})')

    # safe_savgol
    y = np.sin(np.linspace(0, 2 * np.pi, 50))
    ys = safe_savgol(y, window=11, order=3)
    _assert(ys.shape == y.shape, 'savgol same shape')

    ys_short = safe_savgol(np.array([1.0, 2.0, 1.5]), window=11, order=3)
    _assert(len(ys_short) == 3, 'savgol short signal')

    # interpolate_to_grid
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y_vals = np.array([0.0, 1.0, 0.5, 0.0])
    grid = np.linspace(0, 3, 7)
    yi = interpolate_to_grid(x, y_vals, grid)
    _assert(len(yi) == 7, 'interp length')
    _assert(abs(yi[0] - 0.0) < 1e-9, 'interp left boundary')
    _assert(abs(yi[-1] - 0.0) < 1e-9, 'interp right boundary')

    print(f'\nResults: {passed} passed, {failed} failed')
