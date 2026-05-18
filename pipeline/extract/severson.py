"""Severson/MATR Li-ion battery data loader.

Refactored from ``code/severson_pipeline.py``.  All I/O paths are injected as
parameters; there is no global state.

Data format:
  - Cell CSVs: 1000×99 matrices of cumulative discharge Q(V) in Ah
    (1000 voltage points, 3.5→2.0 V; 99 columns = cycles 2-100)
  - Cycle-life CSVs: absolute cycle count to 88 % capacity retention
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from config.settings import PipelineSettings

logger = logging.getLogger(__name__)

# Voltage grid: 1000 points 3.5 → 2.0 V (matches MATLAB Qdlin interpolation)
_V_GRID = np.linspace(3.5, 2.0, 1000)
_DV = abs(_V_GRID[1] - _V_GRID[0])


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_cell_array(cell_path: Path) -> np.ndarray:
    """Load a 1000×99 Q(V) CSV for a single cell.

    Args:
        cell_path: Absolute path to the cell CSV file.

    Returns:
        NumPy array of shape ``(1000, 99)``.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the loaded array has unexpected shape.
    """
    arr = np.loadtxt(str(cell_path), delimiter=",")
    if arr.ndim != 2 or arr.shape[0] != 1000:
        raise ValueError(
            f"Expected (1000, n_cycles) array, got {arr.shape} for {cell_path.name}"
        )
    return arr


def _total_capacity(arr: np.ndarray, col: int) -> float:
    """Total discharge capacity at a given cycle column (last Q(V) value)."""
    return float(arr[-1, col])


def _compute_dqdv(
    q_col: np.ndarray,
    sg_window: int = 11,
    sg_order: int = 3,
) -> np.ndarray:
    """Compute smoothed dQ/dV from a single Q(V) column.

    Args:
        q_col: 1-D capacity array of length 1000.
        sg_window: Savitzky-Golay window length.
        sg_order: Savitzky-Golay polynomial order.

    Returns:
        Smoothed dQ/dV array of length 1000.
    """
    dqdv = np.gradient(q_col, _V_GRID)
    n = len(dqdv)
    w = sg_window
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w >= sg_order + 2 and w <= n:
        return savgol_filter(dqdv, window_length=w, polyorder=sg_order)
    return dqdv


# ---------------------------------------------------------------------------
# Per-cell feature extraction
# ---------------------------------------------------------------------------

def _extract_cell_features(
    cell_path: Path,
    cycle_life_val: float,
    settings: PipelineSettings,
    split: str,
) -> Optional[dict]:
    """Extract features from one Severson cell CSV.

    Args:
        cell_path: Path to the cell's Q(V) CSV.
        cycle_life_val: Raw cycle-life label (at 88 % retention).
        settings: Pipeline configuration.
        split: Dataset split (``'train'``, ``'test1'``, ``'test2'``).

    Returns:
        Feature dict, or ``None`` if the cell fails QC.
    """
    try:
        arr = _load_cell_array(cell_path)
    except (OSError, ValueError) as exc:
        logger.warning("Cannot load cell", extra={"path": str(cell_path), "error": str(exc)})
        return None

    n_cycles = arr.shape[1]
    q_per_cycle = np.array([_total_capacity(arr, c) for c in range(n_cycles)])

    # QC: require 95% of peak capacity threshold
    cell_peak_q = float(np.nanmax(q_per_cycle))
    valid_mask = q_per_cycle >= settings.qc_valid_threshold * cell_peak_q
    n_valid = int(valid_mask.sum())
    if n_valid < 50:
        logger.debug("Cell failed QC", extra={"cell": cell_path.name, "n_valid": n_valid})
        return None

    valid_indices = np.where(valid_mask)[0]

    # Q_first
    q_first = float(q_per_cycle[valid_indices[0]])

    # delta_Q_var (variance of ΔQ over first ~9 valid cycles)
    early_valid = valid_indices[valid_indices <= 8]
    if len(early_valid) >= 3:
        q_early = q_per_cycle[early_valid]
        delta_q_var = float(np.var(np.diff(q_early)))
    else:
        delta_q_var = float("nan")

    # fade_slope (linear fit over first 30 valid cycles)
    early30 = valid_indices[: min(30, len(valid_indices))]
    if len(early30) >= 2:
        x = np.arange(len(early30), dtype=float)
        fade_slope = float(np.polyfit(x, q_per_cycle[early30], 1)[0])
    else:
        fade_slope = float("nan")

    # cycle_life: rescale from 88% to 80% retention via exponential decay
    if np.isfinite(cycle_life_val) and cycle_life_val > 0:
        ratio = np.log(0.80) / np.log(0.88)
        cycle_life_80 = float(cycle_life_val * ratio)
    else:
        cycle_life_80 = float(cycle_life_val)

    raw_curve = q_per_cycle[valid_indices[: min(100, len(valid_indices))]].tolist()
    cell_id = f"severson_{split}_{cell_path.stem}"

    return {
        "cell_id": cell_id,
        "dataset": "severson_matr",
        "chemistry": "LFP",
        "split": split,
        "Q_first": q_first,
        "delta_Q_var": delta_q_var,
        "fade_slope": fade_slope,
        "cycle_life": cycle_life_80,
        "cycle_life_88pct": float(cycle_life_val),
        "n_valid_cycles": n_valid,
        "raw_capacity_curve": json.dumps(raw_curve),
    }


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_severson(
    data_dir: Path,
    settings: PipelineSettings,
) -> pd.DataFrame:
    """Load and preprocess all Severson/MATR Li-ion cells.

    Args:
        data_dir: Root directory containing ``train/``, ``test1/``, ``test2/``
            sub-directories and matching ``*_cycle_lives.csv`` files.
        settings: Pipeline configuration.

    Returns:
        DataFrame with one row per cell that passed QC.  Columns include
        ``cell_id``, ``chemistry``, ``cycle_life``, ``delta_Q_var``, etc.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Severson data directory not found: {data_dir}")

    records: list[dict] = []
    n_total = 0
    n_qc_fail = 0

    for split in ("train", "test1", "test2"):
        cl_path = data_dir / f"{split}_cycle_lives.csv"
        split_dir = data_dir / split

        if not split_dir.exists():
            logger.warning("Split directory missing", extra={"split": split, "path": str(split_dir)})
            continue

        cycle_lives = np.loadtxt(str(cl_path), delimiter=",") if cl_path.exists() else np.array([])
        cell_files = sorted(
            split_dir.glob("*.csv"),
            key=lambda p: int(p.stem.replace("cell", "")) if p.stem.startswith("cell") else 0,
        )

        if len(cell_files) != len(cycle_lives):
            logger.warning(
                "Cell/cycle-life count mismatch",
                extra={"split": split, "n_cells": len(cell_files), "n_lives": len(cycle_lives)},
            )

        for i, cell_path in enumerate(cell_files):
            n_total += 1
            cl_val = float(cycle_lives[i]) if i < len(cycle_lives) else float("nan")
            rec = _extract_cell_features(cell_path, cl_val, settings, split)
            if rec is None:
                n_qc_fail += 1
            else:
                records.append(rec)

    logger.info(
        "Severson loading complete",
        extra={"n_total": n_total, "n_qc_fail": n_qc_fail, "n_passed": len(records)},
    )
    return pd.DataFrame(records)
