"""
Severson/MATR Li-ion battery dataset preprocessing pipeline.

Data source:
  - Cell Q(V) arrays: petermattia/revisit-severson-et-al (GitHub)
    (generated from MATR data at https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)
  - Each cell CSV: 1000×99 matrix, rows=voltage points (3.5→2.0V), cols=cycles 2-100
    Values are cumulative discharge capacity Q(V) in Ah
  - Cycle lives CSVs: absolute cycle count to 88% capacity retention

Features extracted:
  - Q_first        : total discharge capacity at first available cycle (cycle 2)
  - delta_Q_var    : variance of ΔQ_n = Q_{n+1} - Q_n for n in cycles 2-10
  - IR_slope       : NaN — internal resistance not available in this format
  - fade_slope     : linear slope of total Q over cycles 2-31 (30 cycles)
  - dQdV_peak_pos  : voltage of main dQ/dV peak at cycle 5 (SG-smoothed)
  - cycle_life     : total cycles until Q < 80% of Q_first (from cycle_lives CSV,
                     rescaled from 88% threshold to 80% using decay-rate correction)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path('/Users/melodysong/code/phd/battery_ml/data/li_ion/severson')
PROCESSED_DIR = Path('/Users/melodysong/code/phd/battery_ml/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_OUT = PROCESSED_DIR / 'li_ion_severson.parquet'
SUMMARY_OUT = PROCESSED_DIR / 'severson_summary.json'

# Voltage grid: 1000 points from 3.5V to 2.0V (matches MATLAB Qdlin interpolation)
V_MIN, V_MAX = 2.0, 3.5
N_V = 1000
V_GRID = np.linspace(V_MAX, V_MIN, N_V)  # discharge: high→low voltage
DV = abs(V_GRID[1] - V_GRID[0])          # voltage step ≈ 0.0015 V

NOMINAL_Q = 1.1  # Ah (nominal capacity of LFP cells in Severson dataset)
# QC uses 95% of each cell's own peak capacity (relative), not fixed NOMINAL_Q


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cell_array(split: str, cell_name: str) -> np.ndarray:
    """Load 1000×99 Q(V) array for a cell. Rows=voltage, cols=cycles 2-100."""
    path = DATA_DIR / split / cell_name
    return np.loadtxt(str(path), delimiter=',')


def total_capacity(arr: np.ndarray, col: int) -> float:
    """Total discharge capacity at cycle (col+2) = last Q(V) value in column."""
    # Qdlin is cumulative Q from start of discharge; last value = total Q
    return float(arr[-1, col])


def compute_dqdv(q_col: np.ndarray, sg_window: int = 11, sg_order: int = 3) -> np.ndarray:
    """
    Compute dQ/dV from a Q(V) column using finite differences + Savitzky-Golay filter.
    V_GRID is monotonically decreasing (discharge direction), so we flip sign.
    """
    # dQ/dV = (dQ/dV_index) / (dV/dV_index) = diff(Q) / diff(V)
    dq = np.gradient(q_col, V_GRID)  # dQ/dV at each voltage point
    # Smooth with Savitzky-Golay
    if len(dq) >= sg_window:
        dq_smooth = savgol_filter(dq, window_length=sg_window, polyorder=sg_order)
    else:
        dq_smooth = dq
    return dq_smooth  # units: Ah/V


def dqdv_peak_voltage(q_col: np.ndarray) -> float:
    """Voltage of main dQ/dV peak (most negative dQ/dV since Q increases as V drops)."""
    dqdv = compute_dqdv(q_col)
    # Discharge dQ/dV peak: most negative gradient in the middle voltage range
    # Focus on 2.5-3.3V range (typical LFP main plateau)
    mask = (V_GRID >= 2.5) & (V_GRID <= 3.3)
    if mask.sum() < 5:
        mask = np.ones(len(V_GRID), dtype=bool)
    dqdv_region = dqdv[mask]
    v_region = V_GRID[mask]
    peak_idx = np.argmin(dqdv_region)  # most negative = biggest capacity delivery
    return float(v_region[peak_idx])


def extract_features(split: str, cell_name: str, cycle_life_val: float) -> dict | None:
    """
    Extract all features for one cell.

    Returns None if the cell fails QC.
    """
    try:
        arr = load_cell_array(split, cell_name)
    except Exception as e:
        warnings.warn(f"Cannot load {split}/{cell_name}: {e}")
        return None

    n_cycles = arr.shape[1]  # cycles 2..n_cycles+1

    # ------------------------------------------------------------------
    # QC: must have at least 50 valid cycles
    # ------------------------------------------------------------------
    # Compute total Q per cycle
    q_per_cycle = np.array([total_capacity(arr, c) for c in range(n_cycles)])

    # Use 95% of cell's own peak capacity as threshold (relative QC).
    # This accounts for cell-to-cell variation in nominal capacity.
    cell_peak_q = float(np.nanmax(q_per_cycle))
    valid_mask = q_per_cycle >= 0.95 * cell_peak_q
    n_valid = valid_mask.sum()
    if n_valid < 50:
        return None  # QC fail

    # Map valid cycle indices
    valid_indices = np.where(valid_mask)[0]

    # ------------------------------------------------------------------
    # Feature 1: Q_first = capacity at first valid cycle
    # ------------------------------------------------------------------
    q_first = float(q_per_cycle[valid_indices[0]])

    # ------------------------------------------------------------------
    # Feature 2: delta_Q_var = Var(ΔQ) for cycles 2-10 (indices 0-8 = cycles 2-10)
    # We use the first 9 transitions among valid cycles, up to index 8
    # ------------------------------------------------------------------
    early_valid = valid_indices[valid_indices <= 8]  # within first 9 cycles (2-10)
    if len(early_valid) >= 3:
        q_early = q_per_cycle[early_valid]
        delta_q = np.diff(q_early)
        delta_q_var = float(np.var(delta_q))
    else:
        delta_q_var = np.nan

    # ------------------------------------------------------------------
    # Feature 3: IR_slope — not available in Q(V) format
    # ------------------------------------------------------------------
    ir_slope = np.nan

    # ------------------------------------------------------------------
    # Feature 4: fade_slope = linear slope of Q over first 30 valid cycles
    # ------------------------------------------------------------------
    early30 = valid_indices[:min(30, len(valid_indices))]
    if len(early30) >= 2:
        x = np.arange(len(early30), dtype=float)
        y = q_per_cycle[early30]
        fade_slope = float(np.polyfit(x, y, 1)[0])
    else:
        fade_slope = np.nan

    # ------------------------------------------------------------------
    # Feature 5: dQdV_peak_pos = voltage of main dQ/dV peak at cycle 5
    # Cycle 5 corresponds to column index 3 (col 0=cycle2, col 3=cycle5)
    # ------------------------------------------------------------------
    cycle5_col = 3  # cycle 5 = column index 3
    if cycle5_col < n_cycles and valid_mask[cycle5_col]:
        dqdv_peak = dqdv_peak_voltage(arr[:, cycle5_col])
    elif len(valid_indices) >= 4:
        dqdv_peak = dqdv_peak_voltage(arr[:, valid_indices[3]])
    else:
        dqdv_peak = np.nan

    # ------------------------------------------------------------------
    # Feature 6: cycle_life
    # The cycle_lives CSV uses 88% retention threshold (0.88 Ah cutoff for 1.1Ah cells).
    # We rescale to 80% retention using exponential decay approximation:
    #   Q(n) ≈ Q_first * exp(-lambda * n)
    #   At 88%: n_88 = -ln(0.88)/lambda → lambda = -ln(0.88)/n_88
    #   At 80%: n_80 = -ln(0.80)/lambda = n_88 * ln(0.80)/ln(0.88)
    # ------------------------------------------------------------------
    if not np.isnan(cycle_life_val) and cycle_life_val > 0:
        ratio = np.log(0.80) / np.log(0.88)
        cycle_life_80 = float(cycle_life_val * ratio)
    else:
        cycle_life_80 = float(cycle_life_val)

    # ------------------------------------------------------------------
    # raw_capacity_curve: Q per valid cycle as JSON list
    # ------------------------------------------------------------------
    raw_curve = q_per_cycle[valid_indices[:min(100, len(valid_indices))]].tolist()

    cell_id = f"severson_{split}_{cell_name.replace('.csv', '')}"

    return {
        'cell_id': cell_id,
        'dataset': 'severson_matr',
        'chemistry': 'LFP',
        'split': split,
        'Q_first': q_first,
        'delta_Q_var': delta_q_var,
        'IR_slope': ir_slope,
        'fade_slope': fade_slope,
        'dQdV_peak_pos': dqdv_peak,
        'cycle_life': cycle_life_80,
        'cycle_life_88pct': float(cycle_life_val),
        'n_valid_cycles': int(n_valid),
        'raw_capacity_curve': json.dumps(raw_curve),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Severson/MATR Li-ion Battery Preprocessing Pipeline")
    print("=" * 60)

    issues = []
    all_records = []
    n_cells_raw = 0
    n_qc_fail = 0

    for split in ['train', 'test1', 'test2']:
        # Load cycle lives
        cl_path = DATA_DIR / f'{split}_cycle_lives.csv'
        cycle_lives = np.loadtxt(str(cl_path), delimiter=',')

        # Get cell file list (sorted for reproducibility)
        split_dir = DATA_DIR / split
        cell_files = sorted(split_dir.glob('*.csv'),
                            key=lambda p: int(p.stem.replace('cell', '')))

        if len(cell_files) != len(cycle_lives):
            msg = (f"Mismatch: {split} has {len(cell_files)} CSVs "
                   f"but {len(cycle_lives)} cycle lives")
            warnings.warn(msg)
            issues.append(msg)

        print(f"\n[{split.upper()}] {len(cell_files)} cells")

        for i, cell_path in enumerate(cell_files):
            n_cells_raw += 1
            cl_val = float(cycle_lives[i]) if i < len(cycle_lives) else np.nan

            rec = extract_features(split, cell_path.name, cl_val)
            if rec is None:
                n_qc_fail += 1
                print(f"  FAIL QC: {cell_path.name}")
            else:
                all_records.append(rec)
                print(f"  OK  {rec['cell_id']:40s}  "
                      f"Q_first={rec['Q_first']:.3f}  "
                      f"cycle_life={rec['cycle_life']:.0f}")

    n_after_qc = len(all_records)
    print(f"\n{'='*60}")
    print(f"Raw cells:       {n_cells_raw}")
    print(f"QC failures:     {n_qc_fail}")
    print(f"After QC:        {n_after_qc}")

    if n_after_qc == 0:
        print("ERROR: No cells passed QC. Check data.")
        return

    # ------------------------------------------------------------------
    # Build DataFrame and save to Parquet
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_records)
    df.to_parquet(str(PARQUET_OUT), index=False)
    print(f"\nSaved parquet: {PARQUET_OUT}")
    print(f"Columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    cycle_lives_all = df['cycle_life'].dropna()
    median_cl = float(np.median(cycle_lives_all)) if len(cycle_lives_all) else np.nan

    features_computed = [
        f for f in ['Q_first', 'delta_Q_var', 'IR_slope', 'fade_slope', 'dQdV_peak_pos']
        if df[f].notna().any()
    ]
    features_nan = [
        f for f in ['Q_first', 'delta_Q_var', 'IR_slope', 'fade_slope', 'dQdV_peak_pos']
        if df[f].isna().all()
    ]
    if features_nan:
        issues.append(f"All-NaN features (not available in data format): {features_nan}")

    summary = {
        "n_cells_raw": n_cells_raw,
        "n_cells_after_qc": n_after_qc,
        "n_qc_failures": n_qc_fail,
        "median_cycle_life": round(median_cl, 1),
        "median_cycle_life_88pct": round(
            float(np.median(df['cycle_life_88pct'].dropna())), 1),
        "features_computed": features_computed,
        "features_unavailable": features_nan,
        "issues": issues,
        "download_method": (
            "Data sourced from petermattia/revisit-severson-et-al (GitHub), which "
            "contains Q(V) arrays (1000 voltage pts × 99 cycles, cycles 2-100) and "
            "cycle-life labels derived from the original MATR/Severson dataset "
            "(https://data.matr.io/1/projects/5c48dd2bc625d700019f3204). "
            "The MATR website's React SPA has a non-functional API "
            "(REACT_APP_API_URL placeholder was never substituted at build time), "
            "making direct download impossible. The petermattia repo was generated "
            "from the original MATLAB structs by the paper co-author Peter Attia."
        ),
        "data_format_note": (
            "Cell CSVs: 1000×99 matrices of cumulative discharge Q(V) in Ah, "
            "linearly interpolated to 1000 voltage points (3.5→2.0V). "
            "IR_slope is NaN because internal resistance is not available "
            "in this data format (requires raw time-series from MATR pkl files). "
            "cycle_life (80%) estimated from cycle_life_88pct via exponential "
            "decay rescaling: n_80 = n_88 × ln(0.80)/ln(0.88) ≈ n_88 × 1.62."
        ),
        "splits": {
            "train": int((df['split'] == 'train').sum()),
            "test1": int((df['split'] == 'test1').sum()),
            "test2": int((df['split'] == 'test2').sum()),
        },
    }

    with open(str(SUMMARY_OUT), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {SUMMARY_OUT}")

    # Print feature stats
    print("\nFeature statistics:")
    for feat in ['Q_first', 'delta_Q_var', 'fade_slope', 'dQdV_peak_pos', 'cycle_life']:
        col = df[feat].dropna()
        if len(col):
            print(f"  {feat:20s}: n={len(col):3d}  "
                  f"mean={col.mean():.4f}  std={col.std():.4f}  "
                  f"[{col.min():.4f}, {col.max():.4f}]")
        else:
            print(f"  {feat:20s}: ALL NaN")

    print("\nDone.")
    return df, summary


if __name__ == '__main__':
    main()
