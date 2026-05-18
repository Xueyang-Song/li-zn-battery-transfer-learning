"""
FPCA feature extraction from activation-aligned discharge voltage curves.
"""
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA

from utils import interpolate_to_grid, safe_savgol, uniform_voltage_grid


# ---------------------------------------------------------------------------
# dQ/dV computation
# ---------------------------------------------------------------------------

def compute_dqdv(
    voltage: np.ndarray,
    capacity: np.ndarray,
    sg_window: int = 11,
    sg_order: int = 3,
    n_points: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dQ/dV from a discharge voltage-capacity curve.

    Processing steps:
    1. Remove duplicate voltage values (keep first occurrence).
    2. Ensure voltage is sorted in descending order (discharge direction).
    3. Interpolate capacity to a uniform voltage grid of ``n_points`` points.
    4. Compute dQ/dV via central finite differences.
    5. Apply Savitzky-Golay smoothing.

    Parameters
    ----------
    voltage : measured voltage values (V), length N
    capacity : measured capacity values (Ah or mAh), length N
    sg_window : Savitzky-Golay window length (must be odd, ≥ sg_order+2)
    sg_order : Savitzky-Golay polynomial order
    n_points : number of points in the uniform voltage grid

    Returns
    -------
    voltage_grid : uniform voltage array, shape (n_points,)
    dQdV_values : smoothed dQ/dV array, shape (n_points,)

    Notes
    -----
    * Values outside the measured voltage range are extrapolated as boundary
      values (no NaN).
    * If the curve has fewer than sg_order+4 points, SG smoothing is skipped.
    """
    voltage = np.asarray(voltage, dtype=float)
    capacity = np.asarray(capacity, dtype=float)

    if len(voltage) != len(capacity):
        raise ValueError("voltage and capacity must have the same length.")
    if len(voltage) < 4:
        raise ValueError("Need at least 4 data points to compute dQ/dV.")

    # Sort by voltage (ascending for interpolation, then reverse for dQ/dV sign)
    sort_idx = np.argsort(voltage)
    voltage_s = voltage[sort_idx]
    capacity_s = capacity[sort_idx]

    # Remove duplicates
    _, unique_idx = np.unique(voltage_s, return_index=True)
    voltage_s = voltage_s[unique_idx]
    capacity_s = capacity_s[unique_idx]

    v_min, v_max = voltage_s[0], voltage_s[-1]
    if v_min >= v_max:
        raise ValueError(f"Degenerate voltage range: [{v_min}, {v_max}].")

    # Uniform grid (ascending voltage, standard electrochemical convention)
    v_grid = uniform_voltage_grid(v_min, v_max, n_points)

    # Interpolate capacity onto grid
    q_grid = np.interp(v_grid, voltage_s, capacity_s)

    # dQ/dV via central differences
    dv = v_grid[1] - v_grid[0]
    dqdv = np.gradient(q_grid, dv)

    # Smooth
    dqdv_smooth = safe_savgol(dqdv, window=sg_window, order=sg_order)

    return v_grid, dqdv_smooth


# ---------------------------------------------------------------------------
# FPCA fitting
# ---------------------------------------------------------------------------

def fit_fpca(
    curves: List[np.ndarray],
    voltage_grid: np.ndarray,
    n_components: int = 5,
    chemistry: str = 'li_ion'
) -> Tuple[FPCA, np.ndarray]:
    """
    Fit FPCA to a collection of dQ/dV curves defined on a common voltage grid.

    Parameters
    ----------
    curves : list of 1-D arrays, each of length len(voltage_grid)
    voltage_grid : common voltage axis for all curves
    n_components : number of FPCA components to retain
    chemistry : informational label (used only in error messages)

    Returns
    -------
    fpca : fitted skfda FPCA object
    scores : np.ndarray of shape (n_cells, n_components)

    Raises
    ------
    ValueError if fewer cells than n_components are provided.
    """
    n_cells = len(curves)
    if n_cells == 0:
        raise ValueError(f"No curves provided for chemistry={chemistry}.")
    if n_cells < n_components:
        warnings.warn(
            f"chemistry={chemistry}: only {n_cells} cells but n_components={n_components}. "
            f"Reducing to {n_cells} components.",
            UserWarning
        )
        n_components = n_cells

    # Stack into matrix: shape (n_cells, n_grid_points)
    matrix = np.stack([np.asarray(c, dtype=float) for c in curves], axis=0)

    # Replace NaN/Inf with row mean
    for i in range(len(matrix)):
        bad = ~np.isfinite(matrix[i])
        if bad.any():
            row_mean = np.nanmean(matrix[i]) if not np.all(bad) else 0.0
            matrix[i][bad] = row_mean

    fd = FDataGrid(data_matrix=matrix, grid_points=voltage_grid)

    fpca = FPCA(n_components=n_components)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scores = fpca.fit_transform(fd)

    return fpca, np.asarray(scores)


# ---------------------------------------------------------------------------
# Cross-chemistry alignment
# ---------------------------------------------------------------------------

def align_fpca_bases(
    li_fpca: FPCA,
    zn_fpca: FPCA,
    li_voltage_grid: np.ndarray,
    zn_voltage_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project both chemistries' FPCA components onto a common normalized voltage
    axis [0, 1] so that Li-ion and Zn-ion features live in the same space.

    Normalization: V_norm = (V - V_min) / (V_max - V_min) per chemistry.
    The FPCA *basis functions* are then interpolated onto ``n_common`` evenly
    spaced points on [0, 1] and returned as matrices.

    Parameters
    ----------
    li_fpca : fitted FPCA object for Li-ion dQ/dV curves
    zn_fpca : fitted FPCA object for Zn-ion dQ/dV curves
    li_voltage_grid : original voltage grid used for Li-ion FPCA
    zn_voltage_grid : original voltage grid used for Zn-ion FPCA

    Returns
    -------
    li_basis_aligned : np.ndarray, shape (n_components_li, n_common)
    zn_basis_aligned : np.ndarray, shape (n_components_zn, n_common)

    Notes
    -----
    These aligned basis matrices can be used to project new dQ/dV curves
    (expressed on the common [0,1] axis) into each chemistry's FPCA space.
    """
    n_common = 500

    def _normalize_grid(v: np.ndarray) -> np.ndarray:
        vmin, vmax = v.min(), v.max()
        if vmax == vmin:
            return np.zeros_like(v)
        return (v - vmin) / (vmax - vmin)

    common = np.linspace(0.0, 1.0, n_common)

    def _align_components(fpca_obj: FPCA, v_grid: np.ndarray) -> np.ndarray:
        v_norm = _normalize_grid(v_grid)
        # skfda FPCA components: shape (n_components, n_grid)
        components = fpca_obj.components_.data_matrix[:, :, 0]  # (K, n_grid)
        aligned = np.zeros((components.shape[0], n_common))
        for k in range(components.shape[0]):
            aligned[k] = interpolate_to_grid(v_norm, components[k], common)
        return aligned

    li_basis_aligned = _align_components(li_fpca, li_voltage_grid)
    zn_basis_aligned = _align_components(zn_fpca, zn_voltage_grid)

    return li_basis_aligned, zn_basis_aligned


# ---------------------------------------------------------------------------
# Full feature extraction
# ---------------------------------------------------------------------------

def extract_all_features(
    cell_data: pd.DataFrame,
    fpca_li: FPCA,
    fpca_zn: Optional[FPCA],
    n_components: int = 3
) -> pd.DataFrame:
    """
    Extract the full feature matrix used for model training.

    Expected ``cell_data`` columns
    --------------------------------
    * cell_id : str
    * chemistry : 'li_ion' | 'zn_ion'
    * Q_norm_act : float  — capacity at N_act (normalised to 1.0 at N_act)
    * fpca_scores : object — np.ndarray of length ≥ n_components, or None
    * cycle_life : int (optional, used as target, not feature)
    * delta_Q_var : float (optional scalar feature: variance of ΔQ over first
                          cycles; if absent, computed as NaN)

    Returns
    -------
    DataFrame with columns:
        cell_id, chemistry,
        fpca_1 .. fpca_{n_components},
        Q_norm_act,
        delta_Q_var,
        [cycle_life]  — preserved if present in cell_data
    """
    records = []
    for _, row in cell_data.iterrows():
        cid = row['cell_id']
        chem = row.get('chemistry', 'unknown')
        q_act = float(row.get('Q_norm_act', np.nan))
        dq_var = float(row.get('delta_Q_var', np.nan))

        # FPCA scores
        raw_scores = row.get('fpca_scores', None)
        if raw_scores is not None:
            scores_arr = np.asarray(raw_scores, dtype=float)
            # Pad or truncate to n_components
            if len(scores_arr) >= n_components:
                fpca_vals = scores_arr[:n_components].tolist()
            else:
                fpca_vals = list(scores_arr) + [0.0] * (n_components - len(scores_arr))
        else:
            fpca_vals = [np.nan] * n_components

        record = {'cell_id': cid, 'chemistry': chem}
        for k, v in enumerate(fpca_vals, start=1):
            record[f'fpca_{k}'] = float(v)
        record['Q_norm_act'] = q_act
        record['delta_Q_var'] = dq_var

        # Preserve optional target column
        if 'cycle_life' in row.index:
            record['cycle_life'] = row['cycle_life']

        records.append(record)

    feature_df = pd.DataFrame(records)
    return feature_df


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

    rng = np.random.default_rng(42)
    print('=== fpca_features.py unit tests ===')

    # ---- compute_dqdv -------------------------------------------------------
    # Synthetic discharge curve: linear voltage drop, capacity increases then plateaus
    v = np.linspace(4.2, 2.5, 200)
    q = np.linspace(0.0, 2.8, 200) + 0.02 * rng.normal(size=200)

    v_grid, dqdv = compute_dqdv(v, q, sg_window=11, sg_order=3, n_points=300)
    _assert(len(v_grid) == 300, f'dqdv grid length=300 (got {len(v_grid)})')
    _assert(np.all(np.isfinite(dqdv)), 'dqdv finite')
    _assert(abs(v_grid[0] - v.min()) < 1e-6, 'grid starts at v_min')
    _assert(abs(v_grid[-1] - v.max()) < 1e-6, 'grid ends at v_max')

    # Short curve edge case
    try:
        compute_dqdv(np.array([4.0, 3.5]), np.array([0.0, 1.0]), n_points=10)
        _assert(False, 'too-short curve raises')
    except ValueError:
        _assert(True, 'too-short curve raises')

    # ---- fit_fpca -----------------------------------------------------------
    n_cells = 15
    n_grid = 200
    v_common = np.linspace(2.5, 4.2, n_grid)
    # Synthetic dQ/dV curves with two Gaussian peaks
    def _synthetic_dqdv(rng_):
        peaks = rng_.normal(3.4, 0.05) * np.exp(-((v_common - 3.4) ** 2) / 0.02)
        peaks += rng_.normal(3.8, 0.03) * np.exp(-((v_common - 3.8) ** 2) / 0.01)
        return peaks + 0.01 * rng_.normal(size=n_grid)

    li_curves = [_synthetic_dqdv(rng) for _ in range(n_cells)]
    fpca_obj, scores = fit_fpca(li_curves, v_common, n_components=3, chemistry='li_ion')

    _assert(scores.shape == (n_cells, 3), f'scores shape {scores.shape}')
    _assert(np.all(np.isfinite(scores)), 'scores finite')

    # too few cells → warning + reduced components
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter('always')
        _, scores_few = fit_fpca(li_curves[:2], v_common, n_components=5)
        _assert(len(w_list) >= 1, 'UserWarning when n_cells < n_components')
    _assert(scores_few.shape[1] <= 2, 'components capped at n_cells')

    # ---- align_fpca_bases ---------------------------------------------------
    zn_curves = [_synthetic_dqdv(rng) for _ in range(n_cells)]
    v_zn = np.linspace(0.8, 1.9, n_grid)
    fpca_zn_obj, _ = fit_fpca(zn_curves, v_zn, n_components=3, chemistry='zn_ion')

    li_basis, zn_basis = align_fpca_bases(fpca_obj, fpca_zn_obj, v_common, v_zn)
    _assert(li_basis.shape[0] == 3 and li_basis.shape[1] == 500,
            f'li_basis shape {li_basis.shape}')
    _assert(zn_basis.shape[0] == 3 and zn_basis.shape[1] == 500,
            f'zn_basis shape {zn_basis.shape}')
    _assert(np.all(np.isfinite(li_basis)), 'li_basis finite')

    # ---- extract_all_features -----------------------------------------------
    import warnings as _w
    _w.filterwarnings('ignore')

    n_feat_cells = 20
    rows = []
    for i in range(n_feat_cells):
        rows.append({
            'cell_id': f'cell_{i:03d}',
            'chemistry': 'li_ion' if i < 10 else 'zn_ion',
            'Q_norm_act': rng.uniform(0.9, 1.1),
            'fpca_scores': rng.normal(size=5),
            'delta_Q_var': rng.uniform(0.001, 0.05),
            'cycle_life': rng.integers(200, 1500),
        })
    df_in = pd.DataFrame(rows)
    feat_df = extract_all_features(df_in, fpca_li=fpca_obj, fpca_zn=fpca_zn_obj,
                                   n_components=3)

    _assert(len(feat_df) == n_feat_cells, f'feature rows={n_feat_cells}')
    _assert('fpca_1' in feat_df.columns, 'fpca_1 column present')
    _assert('fpca_3' in feat_df.columns, 'fpca_3 column present')
    _assert('Q_norm_act' in feat_df.columns, 'Q_norm_act column')
    _assert('cycle_life' in feat_df.columns, 'cycle_life preserved')
    _assert(not feat_df['fpca_1'].isna().any(), 'no NaN in fpca_1')

    # Cell with no FPCA scores → NaN
    rows_nan = [{'cell_id': 'x', 'chemistry': 'li_ion',
                 'Q_norm_act': 1.0, 'fpca_scores': None, 'delta_Q_var': 0.0}]
    df_nan = pd.DataFrame(rows_nan)
    feat_nan = extract_all_features(df_nan, fpca_li=fpca_obj, fpca_zn=None,
                                    n_components=3)
    _assert(np.isnan(feat_nan['fpca_1'].iloc[0]), 'None scores → NaN')

    print(f'\nResults: {passed} passed, {failed} failed')
