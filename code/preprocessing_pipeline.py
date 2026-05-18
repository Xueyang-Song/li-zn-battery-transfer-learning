"""
End-to-end preprocessing pipeline for battery ML paper.
Runs all 10 preprocessing steps from raw data to feature matrix.

Step summary
------------
 1. Load and validate raw parquet files
 2. Hold-out Zn-ion test cells (n_test_cells, never touched during training)
 3. Detect activation period (PELT) for each Zn-ion cell
 4. Normalize post-activation capacity; compute Q_norm_act
 5. Run PELT visual validation on random sample
 6. Extract dQ/dV curves for Li-ion cells
 7. Fit FPCA on Li-ion dQ/dV curves
 8. Extract dQ/dV curves for Zn-ion training cells; compute Zn FPCA scores
 9. Build unified feature matrix (Li-ion + Zn-ion train cells)
10. Save feature matrix and pipeline metadata
"""
import json
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = Path('/Users/melodysong/code/phd/battery_ml/')

# Local modules
import sys
sys.path.insert(0, str(BASE_DIR / 'code'))
from pelt_activation import detect_activation_end, normalize_by_activation_end, validate_pelt_sample
from fpca_features import compute_dqdv, fit_fpca, extract_all_features
from utils import capacity_to_soh, find_cycle_life, uniform_voltage_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_parquet(path: str, label: str) -> pd.DataFrame:
    """Load a parquet file or return empty DataFrame with a warning."""
    p = Path(path)
    if not p.exists():
        warnings.warn(f"[Step 1] {label} not found at {path}. Using empty DataFrame.")
        return pd.DataFrame()
    df = pd.read_parquet(p)
    print(f"  Loaded {label}: {len(df)} rows, columns: {list(df.columns)}")
    return df


def _required_columns(df: pd.DataFrame, cols: list, label: str) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _compute_scalar_features(curve: np.ndarray, soh: np.ndarray) -> dict:
    """
    Compute scalar fallback features from a normalized capacity curve.

    Returns dict with:
    - delta_Q_var  : variance of first-order differences (ΔQ) over first 10 cycles
    - cycle_life_80: cycle life at 80 % SOH
    - slope_early  : linear slope of first 20 % of SOH curve
    """
    delta_q = np.diff(curve[:min(10, len(curve))])
    dq_var = float(np.var(delta_q)) if len(delta_q) > 0 else np.nan

    cl80 = find_cycle_life(soh, threshold=0.80)

    n_early = max(2, int(0.2 * len(soh)))
    x_e = np.arange(n_early, dtype=float)
    y_e = soh[:n_early]
    slope_early = float(np.polyfit(x_e, y_e, 1)[0]) if n_early >= 2 else np.nan

    return {
        'delta_Q_var': dq_var,
        'cycle_life_80': cl80,
        'slope_early': slope_early,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    li_ion_parquet_paths: list,
    zn_ion_parquet_path: str,
    output_dir: str = str(BASE_DIR / 'features'),
    test_set_seed: int = 42,
    n_test_cells: int = 10,
    n_fpca_components: int = 3
) -> dict:
    """
    Steps 1–10 of the preprocessing pipeline.

    Parquet schema (expected columns)
    ----------------------------------
    Li-ion parquet:
        cell_id (str), cycle (int), discharge_capacity (float),
        voltage (object/list), capacity_dqdv (object/list)
        [optional: cycle_life (int)]

    Zn-ion parquet:
        cell_id (str), cycle (int), discharge_capacity (float),
        voltage (object/list), capacity_dqdv (object/list)

    Parameters
    ----------
    li_ion_parquet_paths : list of str paths to Li-ion datasets
    zn_ion_parquet_path  : str path to Zn-ion dataset
    output_dir           : directory to write feature_matrix.parquet + metadata
    test_set_seed        : random seed for test-set split
    n_test_cells         : number of Zn-ion cells to hold out for testing
    n_fpca_components    : K FPCA components to retain per chemistry

    Returns
    -------
    dict with keys:
        feature_matrix_path, test_cell_ids, train_cell_ids,
        n_li_cells, n_zn_train_cells, n_zn_test_cells,
        pelt_validation_report
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 – Load and validate raw data
    # ------------------------------------------------------------------
    print('\n[Step 1] Loading parquet files…')
    li_frames = []
    for p in li_ion_parquet_paths:
        df = _load_parquet(p, label=os.path.basename(p))
        if not df.empty:
            li_frames.append(df)

    li_df = pd.concat(li_frames, ignore_index=True) if li_frames else pd.DataFrame()
    zn_df = _load_parquet(zn_ion_parquet_path, label='zn_ion_raw.parquet')

    # Validate columns (only if DataFrames are non-empty)
    REQUIRED = ['cell_id', 'cycle', 'discharge_capacity']
    for df, label in [(li_df, 'Li-ion'), (zn_df, 'Zn-ion')]:
        if not df.empty:
            _required_columns(df, REQUIRED, label)

    # Synthetic fallback: if data absent, generate tiny demo dataset
    if li_df.empty or zn_df.empty:
        print('  WARNING: One or both datasets are empty. Generating synthetic demo data.')
        li_df, zn_df = _make_synthetic_data(seed=test_set_seed)

    # ------------------------------------------------------------------
    # Step 2 – Hold-out Zn-ion test cells
    # ------------------------------------------------------------------
    print('\n[Step 2] Splitting Zn-ion test / train cells…')
    all_zn_ids = list(zn_df['cell_id'].unique())
    rng = np.random.default_rng(test_set_seed)
    rng.shuffle(all_zn_ids)
    n_test = min(n_test_cells, len(all_zn_ids))
    test_cell_ids = all_zn_ids[:n_test]
    train_cell_ids = all_zn_ids[n_test:]
    print(f'  Test: {len(test_cell_ids)} cells  |  Train pool: {len(train_cell_ids)} cells')

    # ------------------------------------------------------------------
    # Step 3 – Detect activation period (PELT) for Zn-ion train cells
    # ------------------------------------------------------------------
    print('\n[Step 3] PELT activation detection for Zn-ion train cells…')
    zn_train_df = zn_df[zn_df['cell_id'].isin(train_cell_ids)].copy()

    # Build per-cell capacity curves (sorted by cycle)
    zn_capacity_curves: dict = {}
    for cid, grp in zn_train_df.sort_values('cycle').groupby('cell_id'):
        zn_capacity_curves[cid] = grp['discharge_capacity'].values.astype(float)

    n_act_results: dict = {}
    for cid, curve in zn_capacity_curves.items():
        n_act, info = detect_activation_end(curve)
        n_act_results[cid] = (n_act, info)

    # ------------------------------------------------------------------
    # Step 4 – Normalize post-activation capacity; compute Q_norm_act
    # ------------------------------------------------------------------
    print('\n[Step 4] Normalizing capacity by Q(N_act)…')
    zn_train_records = []
    for cid, curve in zn_capacity_curves.items():
        n_act, info = n_act_results[cid]
        try:
            post, soh = normalize_by_activation_end(curve, n_act)
        except ValueError as e:
            warnings.warn(f"  Cell {cid}: {e}. Skipping.")
            continue

        scalar = _compute_scalar_features(post, soh)
        zn_train_records.append({
            'cell_id': cid,
            'chemistry': 'zn_ion',
            'Q_norm_act': float(curve[n_act]),
            'n_act': n_act,
            'fpca_scores': None,  # filled in Step 8
            **scalar,
        })

    zn_train_meta = pd.DataFrame(zn_train_records)

    # ------------------------------------------------------------------
    # Step 5 – PELT visual validation (random sample)
    # ------------------------------------------------------------------
    print('\n[Step 5] PELT visual validation…')
    pelt_val_dir = str(BASE_DIR / 'processed/pelt_validation')
    pelt_report = validate_pelt_sample(
        cell_ids=list(zn_capacity_curves.keys()),
        capacity_curves=zn_capacity_curves,
        n_act_results=n_act_results,
        sample_frac=0.10,
        output_dir=pelt_val_dir,
    )
    print(f'  Sampled {pelt_report["n_sampled"]} cells; '
          f'PELT success rate: {pelt_report["pelt_fraction"]:.1%}')

    # ------------------------------------------------------------------
    # Step 6 – Extract dQ/dV curves for Li-ion cells
    # ------------------------------------------------------------------
    print('\n[Step 6] Extracting Li-ion dQ/dV curves…')
    li_dqdv_curves, li_cell_meta = _extract_dqdv_for_chemistry(
        li_df, chemistry='li_ion', n_fpca_components=n_fpca_components
    )

    # ------------------------------------------------------------------
    # Step 7 – Fit FPCA on Li-ion dQ/dV curves
    # ------------------------------------------------------------------
    print('\n[Step 7] Fitting Li-ion FPCA…')
    li_v_grid = None
    fpca_li = None
    li_scores_arr = None

    if li_dqdv_curves:
        curves_list = [c for _, c in li_dqdv_curves]
        li_v_grid = li_dqdv_curves[0][0]  # all share the same grid (after align)
        fpca_li, li_scores_arr = fit_fpca(
            curves_list, li_v_grid,
            n_components=n_fpca_components,
            chemistry='li_ion'
        )
        print(f'  Li-ion FPCA: {len(curves_list)} cells, '
              f'{n_fpca_components} components retained.')
        # Attach scores to meta
        for i, (cid, _) in enumerate(li_dqdv_curves):
            li_cell_meta.loc[li_cell_meta['cell_id'] == cid, 'fpca_scores_obj'] = \
                li_scores_arr[i].tolist()
        # Convert to column of arrays
        li_cell_meta['fpca_scores'] = li_cell_meta.get('fpca_scores_obj', None)
    else:
        print('  No Li-ion dQ/dV data available; skipping FPCA.')

    # ------------------------------------------------------------------
    # Step 8 – Zn-ion dQ/dV + FPCA scores (projected onto Li-ion basis)
    # ------------------------------------------------------------------
    print('\n[Step 8] Extracting Zn-ion dQ/dV and projecting onto Li-ion basis…')
    zn_train_df2 = zn_df[zn_df['cell_id'].isin(train_cell_ids)].copy()
    zn_dqdv_curves, _ = _extract_dqdv_for_chemistry(
        zn_train_df2, chemistry='zn_ion', n_fpca_components=n_fpca_components,
        v_grid_target=li_v_grid  # project onto Li-ion voltage grid if provided
    )

    if zn_dqdv_curves and fpca_li is not None and li_v_grid is not None:
        from skfda import FDataGrid
        zn_curves_list = [c for _, c in zn_dqdv_curves]
        # Project Zn curves onto the *same* Li-ion common voltage grid
        import numpy as _np
        from utils import interpolate_to_grid as _interp
        zn_common_curves = []
        for v_arr, dqdv_arr in zn_dqdv_curves:
            # Interpolate each Zn dQ/dV onto the Li-ion voltage grid
            dqdv_on_li_grid = _interp(v_arr, dqdv_arr, li_v_grid)
            zn_common_curves.append(dqdv_on_li_grid)

        zn_matrix = _np.stack(zn_common_curves, axis=0)
        fd_zn = FDataGrid(data_matrix=zn_matrix, grid_points=li_v_grid)
        zn_scores_arr = fpca_li.transform(fd_zn)
        print(f'  Zn-ion scores projected: {zn_matrix.shape[0]} cells.')

        for i, (cid, _) in enumerate(zn_dqdv_curves):
            mask = zn_train_meta['cell_id'] == cid
            zn_train_meta.loc[mask, 'fpca_scores'] = \
                [zn_scores_arr[i].tolist()] * mask.sum()

    # ------------------------------------------------------------------
    # Step 9 – Build unified feature matrix
    # ------------------------------------------------------------------
    print('\n[Step 9] Building unified feature matrix…')
    # Li-ion feature rows
    li_feat_rows = []
    if li_cell_meta is not None and not li_cell_meta.empty:
        for _, row in li_cell_meta.iterrows():
            scores = row.get('fpca_scores_obj', None) or row.get('fpca_scores', None)
            if isinstance(scores, list):
                scores = np.array(scores)
            li_feat_rows.append({
                'cell_id': row['cell_id'],
                'chemistry': 'li_ion',
                'Q_norm_act': row.get('Q_norm_act', np.nan),
                'fpca_scores': scores,
                'delta_Q_var': row.get('delta_Q_var', np.nan),
                'cycle_life': row.get('cycle_life', np.nan),
            })
    li_feat_df = pd.DataFrame(li_feat_rows)

    # Zn-ion feature rows (train only)
    zn_feat_rows = []
    for _, row in zn_train_meta.iterrows():
        scores = row.get('fpca_scores', None)
        if isinstance(scores, list):
            scores = np.array(scores)
        zn_feat_rows.append({
            'cell_id': row['cell_id'],
            'chemistry': 'zn_ion',
            'Q_norm_act': row.get('Q_norm_act', np.nan),
            'fpca_scores': scores,
            'delta_Q_var': row.get('delta_Q_var', np.nan),
            'cycle_life': row.get('cycle_life', np.nan),
        })
    zn_feat_df = pd.DataFrame(zn_feat_rows)

    all_feat_df = pd.concat([li_feat_df, zn_feat_df], ignore_index=True)

    # Expand FPCA scores to individual columns
    for k in range(1, n_fpca_components + 1):
        all_feat_df[f'fpca_{k}'] = all_feat_df['fpca_scores'].apply(
            lambda s, _k=k: (
                float(np.asarray(s)[_k - 1])
                if s is not None and not (
                    isinstance(s, float) and np.isnan(s)
                ) and hasattr(s, '__len__') and len(s) >= _k
                else np.nan
            )
        )
    all_feat_df = all_feat_df.drop(columns=['fpca_scores'], errors='ignore')

    # ------------------------------------------------------------------
    # Step 10 – Save feature matrix and metadata
    # ------------------------------------------------------------------
    print('\n[Step 10] Saving outputs…')
    feat_path = os.path.join(output_dir, 'feature_matrix.parquet')
    all_feat_df.to_parquet(feat_path, index=False)
    print(f'  Feature matrix saved: {feat_path}  ({len(all_feat_df)} rows)')

    metadata = {
        'feature_matrix_path': feat_path,
        'test_cell_ids': [str(x) for x in test_cell_ids],
        'train_cell_ids': [str(x) for x in train_cell_ids],
        'n_li_cells': int(li_df['cell_id'].nunique()) if not li_df.empty else 0,
        'n_zn_train_cells': len(train_cell_ids),
        'n_zn_test_cells': len(test_cell_ids),
        'pelt_validation_report': pelt_report,
    }

    meta_path = os.path.join(output_dir, 'pipeline_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f'  Metadata saved: {meta_path}')

    return metadata


# ---------------------------------------------------------------------------
# Internal helper: dQ/dV extraction per chemistry
# ---------------------------------------------------------------------------

def _extract_dqdv_for_chemistry(
    df: pd.DataFrame,
    chemistry: str,
    n_fpca_components: int = 3,
    reference_cycle: int = 5,
    n_points: int = 300,
    v_grid_target: Optional[np.ndarray] = None,
) -> tuple:
    """
    For each cell in ``df``, pick a representative cycle near ``reference_cycle``
    and compute its dQ/dV curve on a common voltage grid.

    Returns
    -------
    dqdv_list : list of (v_grid, dqdv_array) tuples — one per cell
    meta_df   : DataFrame with cell-level metadata (cell_id, Q_norm_act, …)
    """
    from fpca_features import compute_dqdv as _compute_dqdv

    # Voltage column names vary across datasets
    v_col = next((c for c in ('voltage', 'voltage_V', 'V') if c in df.columns), None)
    q_col = next((c for c in ('capacity_dqdv', 'capacity', 'discharge_capacity')
                  if c in df.columns), None)

    dqdv_list = []
    meta_rows = []

    for cid, grp in df.sort_values('cycle').groupby('cell_id'):
        cap_curve = grp['discharge_capacity'].values.astype(float)
        q_act = float(cap_curve[0]) if len(cap_curve) > 0 else np.nan

        # Pick representative cycle
        avail_cycles = grp['cycle'].values
        ref_idx = np.argmin(np.abs(avail_cycles - reference_cycle))
        row_ref = grp.iloc[ref_idx]

        v_arr, q_arr = None, None
        if v_col and q_col:
            raw_v = row_ref[v_col]
            raw_q = row_ref[q_col]
            # Data may be stored as list/array serialised in parquet
            if isinstance(raw_v, (list, np.ndarray)) and len(raw_v) >= 4:
                v_arr = np.asarray(raw_v, dtype=float)
                q_arr = np.asarray(raw_q, dtype=float)

        if v_arr is not None and q_arr is not None:
            try:
                v_grid, dqdv = _compute_dqdv(v_arr, q_arr, n_points=n_points)
                if v_grid_target is not None:
                    from utils import interpolate_to_grid
                    dqdv = interpolate_to_grid(v_grid, dqdv, v_grid_target)
                    v_grid = v_grid_target
                dqdv_list.append((v_grid, dqdv))
            except (ValueError, Exception):
                pass

        # Scalar features
        from utils import capacity_to_soh, find_cycle_life
        soh = capacity_to_soh(cap_curve) if len(cap_curve) > 0 else np.array([])
        delta_q = np.diff(cap_curve[:10]) if len(cap_curve) > 1 else np.array([])
        dq_var = float(np.var(delta_q)) if len(delta_q) > 0 else np.nan
        cl_80 = find_cycle_life(soh, 0.80) if len(soh) > 0 else len(cap_curve)

        meta_rows.append({
            'cell_id': cid,
            'chemistry': chemistry,
            'Q_norm_act': q_act,
            'delta_Q_var': dq_var,
            'cycle_life': cl_80,
            'fpca_scores': None,
        })

    meta_df = pd.DataFrame(meta_rows)
    return dqdv_list, meta_df


# ---------------------------------------------------------------------------
# Synthetic demo data generator (used when parquet files are absent)
# ---------------------------------------------------------------------------

def _make_synthetic_data(
    n_li: int = 40,
    n_zn: int = 25,
    n_cycles: int = 300,
    seed: int = 42
) -> tuple:
    """
    Generate tiny synthetic Li-ion and Zn-ion DataFrames for smoke-testing.
    Each cell has one row per cycle with discharge_capacity.
    """
    rng = np.random.default_rng(seed)
    rows_li, rows_zn = [], []

    for i in range(n_li):
        cl = rng.integers(150, 600)
        cap = np.maximum(
            0.5,
            1.0 - np.linspace(0, 0.4, n_cycles) + rng.normal(0, 0.01, n_cycles)
        )
        for cyc in range(n_cycles):
            rows_li.append({
                'cell_id': f'li_{i:03d}',
                'cycle': cyc,
                'discharge_capacity': cap[cyc],
            })

    for i in range(n_zn):
        act_len = rng.integers(5, 15)
        act = np.linspace(0.9, 1.15, act_len) + rng.normal(0, 0.01, act_len)
        deg_len = n_cycles - act_len
        deg = np.linspace(1.15, 0.65, deg_len) + rng.normal(0, 0.01, deg_len)
        cap = np.concatenate([act, deg])
        for cyc in range(n_cycles):
            rows_zn.append({
                'cell_id': f'zn_{i:03d}',
                'cycle': cyc,
                'discharge_capacity': cap[cyc],
            })

    return pd.DataFrame(rows_li), pd.DataFrame(rows_zn)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== preprocessing_pipeline.py ===')
    print('Running pipeline with synthetic data (parquet files may not exist)…\n')

    result = run_full_pipeline(
        li_ion_parquet_paths=[
            str(BASE_DIR / 'processed/li_ion_severson.parquet'),
            str(BASE_DIR / 'processed/li_ion_calce.parquet'),
        ],
        zn_ion_parquet_path=str(BASE_DIR / 'processed/zn_ion_raw.parquet'),
    )
    print('\n=== Pipeline result ===')
    # Make result JSON-serialisable (drop non-serialisable objects)
    safe_result = {k: v for k, v in result.items() if k != 'pelt_validation_report'}
    safe_result['pelt_validation_report'] = {
        k: v for k, v in result['pelt_validation_report'].items()
        if k != 'plot_paths'
    }
    print(json.dumps(safe_result, indent=2, default=str))
    print(f"\nTest cell IDs ({len(result['test_cell_ids'])}):",
          result['test_cell_ids'][:5], '…')
    print(f"Feature matrix: {result['feature_matrix_path']}")
