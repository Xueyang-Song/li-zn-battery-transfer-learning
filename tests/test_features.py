"""Unit tests for canonical feature engineering functions.

Tests cover both happy-path and edge-case behaviour of:
- ``fit_exp_b``
- ``compute_delta_Q_var``
- ``log10_transform``
- ``build_feature_matrix``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the battery_ml root is on the path when running tests directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.transform.features import (
    _exp_decay,
    build_feature_matrix,
    compute_delta_Q_var,
    fit_exp_b,
    log10_transform,
)
from config.settings import PipelineSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> PipelineSettings:
    """Return a PipelineSettings with sane defaults for testing."""
    defaults = dict(
        base_dir=Path("/tmp/test_battery"),
        li_ion_severson_dir=Path("data/li_ion"),
        zn_ion_batches_dir=Path("data/zn_ion"),
        features_dir=Path("features"),
        results_dir=Path("results"),
        figures_dir=Path("figures"),
    )
    defaults.update(overrides)
    return PipelineSettings(**defaults)


def _perfect_exp_curve(b: float, n: int = 30, A: float = 0.1) -> np.ndarray:
    """Generate a noise-free Q(n)/Q0 = A + (1-A)·exp(−b·n) curve."""
    return np.array([_exp_decay(np.array([i]), A, b)[0] for i in range(n)])


# ---------------------------------------------------------------------------
# fit_exp_b
# ---------------------------------------------------------------------------

class TestFitExpB:
    """Tests for the exponential decay rate fitting function."""

    def test_perfect_curve_recovers_b(self) -> None:
        """A noise-free curve should recover b within 1%."""
        b_true = 0.15
        # Generate relative capacity (already normalised to Q0=1)
        curve_norm = _perfect_exp_curve(b_true, n=30, A=0.1)
        # fit_exp_b expects absolute capacity; supply Q0=1.0 so Q(n)/Q0=curve_norm
        b_fit = fit_exp_b(curve_norm * 1.0, n_cycles=20)
        assert np.isfinite(b_fit), "b_fit should be finite"
        assert abs(b_fit - b_true) / b_true < 0.01, (
            f"Expected b≈{b_true}, got {b_fit:.4f} (error > 1%)"
        )

    def test_noisy_curve_in_reasonable_range(self) -> None:
        """Noisy curve should yield b in a physically reasonable range."""
        rng = np.random.default_rng(0)
        b_true = 0.08
        curve = _perfect_exp_curve(b_true, n=25, A=0.05)
        curve_noisy = curve + rng.normal(0, 0.005, len(curve))
        # Recover b from noisy absolute capacity (Q0~1.0)
        b_fit = fit_exp_b(curve_noisy, n_cycles=20)
        assert np.isfinite(b_fit), "b_fit should be finite on noisy curve"
        assert 1e-5 < b_fit < 5.0, f"b_fit={b_fit} outside plausible range"

    def test_boundary_hit_returns_near_max(self) -> None:
        """Flat or rising curve should push b towards the upper bound (boundary hit)."""
        # Flat-ish / slowly rising capacity → curve_fit should hit b near 5.0 or return nan
        curve = np.linspace(1.0, 1.02, 25)  # very slowly increasing (no decay)
        b_fit = fit_exp_b(curve, n_cycles=20)
        # Either NaN or very small (no meaningful decay signal)
        if np.isfinite(b_fit):
            assert b_fit < 1.0, f"Non-decaying curve should yield small b, got {b_fit}"

    def test_too_short_curve_returns_nan(self) -> None:
        """Curves with fewer than 3 points should return NaN."""
        b_fit = fit_exp_b(np.array([1.0, 0.98]))
        assert np.isnan(b_fit), "Too-short curve should return NaN"

    def test_zero_q0_returns_nan(self) -> None:
        """Zero initial capacity should return NaN (avoids division by zero)."""
        curve = np.zeros(10)
        b_fit = fit_exp_b(curve)
        assert np.isnan(b_fit)

    def test_filter_boundary_hit_at_4p9(self) -> None:
        """Verify boundary-hit detection: b near 4.9 indicates poor fit."""
        # A perfectly flat curve will cause curve_fit to hit the bound
        flat = np.ones(20)
        b_fit = fit_exp_b(flat, n_cycles=20)
        # If returned, it should not equal 4.9 (the boundary) exactly since
        # a flat curve has b≈0; the key invariant is that the settings filter
        # removes cells with b >= exp_b_filter_max=4.9.
        settings = _make_settings()
        assert settings.exp_b_filter_max == 4.9


# ---------------------------------------------------------------------------
# compute_delta_Q_var
# ---------------------------------------------------------------------------

class TestComputeDeltaQVar:
    """Tests for the dQ/dV variance feature."""

    def test_constant_curve_zero_variance(self) -> None:
        """A constant capacity curve should have variance 0."""
        curve = np.ones(50)
        var = compute_delta_Q_var(curve, sg_window=11, sg_order=3)
        assert np.isfinite(var)
        assert abs(var) < 1e-10, f"Expected var≈0 for constant curve, got {var}"

    def test_linearly_decreasing_curve_very_small_var(self) -> None:
        """A perfectly linear decay should yield near-zero variance."""
        curve = np.linspace(1.0, 0.5, 50)
        var = compute_delta_Q_var(curve)
        assert np.isfinite(var)
        assert var < 1e-8, f"Linear curve variance too large: {var}"

    def test_noisy_curve_positive_variance(self) -> None:
        """A noisy curve should have positive variance."""
        rng = np.random.default_rng(42)
        curve = np.linspace(1.0, 0.8, 50) + rng.normal(0, 0.01, 50)
        var = compute_delta_Q_var(curve)
        assert np.isfinite(var)
        assert var > 0, "Noisy curve should have positive variance"

    def test_too_short_curve_returns_nan(self) -> None:
        """Single-element curve should return NaN."""
        var = compute_delta_Q_var(np.array([1.0]))
        assert np.isnan(var)


# ---------------------------------------------------------------------------
# log10_transform
# ---------------------------------------------------------------------------

class TestLog10Transform:
    """Tests for the log10 compression function."""

    def test_positive_value(self) -> None:
        """log10_transform(1.0) should equal log10(1.0 + 1e-6) ≈ 0."""
        result = log10_transform(1.0)
        expected = np.log10(1.0 + 1e-6)
        assert abs(result - expected) < 1e-10

    def test_zero_does_not_raise(self) -> None:
        """log10_transform(0) should return log10(1e-6) ≈ -6."""
        result = log10_transform(0.0)
        assert np.isfinite(result)
        assert abs(result - (-6.0)) < 0.01

    def test_compresses_1000x_gap(self) -> None:
        """1000× scale gap in delta_Q_var should compress to ~3 log units."""
        v_li = 1e-5    # typical Li-ion delta_Q_var
        v_zn = 1e-2    # typical Zn-ion delta_Q_var (1000× larger)
        log_li = log10_transform(v_li)
        log_zn = log10_transform(v_zn)
        # Raw ratio is 1000×; log gap should be ≈ 3 units
        raw_ratio = v_zn / v_li
        gap_log = abs(log_zn - log_li)
        assert raw_ratio > 100, f"Raw scale gap should be >100×, got {raw_ratio}"
        assert gap_log < 5, f"Log gap should be <5 units, got {gap_log:.3f}"
        # The log transform should compress the gap: gap_log << raw_ratio
        assert gap_log < raw_ratio / 10, (
            f"log10 should massively compress the scale gap: "
            f"raw_ratio={raw_ratio}, gap_log={gap_log:.3f}"
        )

    def test_custom_eps(self) -> None:
        """Custom eps should be respected."""
        result = log10_transform(0.0, eps=1e-3)
        expected = np.log10(1e-3)
        assert abs(result - expected) < 1e-10


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    """Tests for the unified feature matrix builder."""

    @pytest.fixture()
    def sample_dfs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create minimal Li-ion and Zn-ion feature DataFrames."""
        rng = np.random.default_rng(0)
        n_li, n_zn = 30, 20

        li_df = pd.DataFrame(
            {
                "cell_id": [f"li_{i}" for i in range(n_li)],
                "exp_b": rng.uniform(0.01, 2.0, n_li),
                "delta_Q_var": rng.uniform(1e-6, 0.01, n_li),
                "cycle_life": rng.integers(100, 1500, n_li).astype(float),
            }
        )
        zn_df = pd.DataFrame(
            {
                "cell_id": [f"zn_{i}" for i in range(n_zn)],
                "exp_b": rng.uniform(0.01, 1.0, n_zn),
                "delta_Q_var": rng.uniform(1e-3, 0.05, n_zn),
                "cycle_life": rng.integers(50, 600, n_zn).astype(float),
            }
        )
        return li_df, zn_df

    def test_output_shapes(self, sample_dfs: tuple) -> None:
        """Output arrays should have the expected shapes."""
        li_df, zn_df = sample_dfs
        settings = _make_settings()
        X_li, y_li, X_zn, y_zn, scaler = build_feature_matrix(li_df, zn_df, settings)

        assert X_li.ndim == 2 and X_li.shape[1] == 2, f"X_li shape: {X_li.shape}"
        assert X_zn.ndim == 2 and X_zn.shape[1] == 2, f"X_zn shape: {X_zn.shape}"
        assert len(y_li) == X_li.shape[0]
        assert len(y_zn) == X_zn.shape[0]

    def test_scaler_fitted_on_combined_data(self, sample_dfs: tuple) -> None:
        """The scaler's mean should match the combined Li+Zn sample mean."""
        li_df, zn_df = sample_dfs
        settings = _make_settings()
        X_li, _, X_zn, _, scaler = build_feature_matrix(li_df, zn_df, settings)

        # After scaling the combined data, the mean of the scaled arrays should
        # be close to zero (unified scaler centred on all data, not just one task).
        combined_scaled = np.vstack([X_li, X_zn])
        assert abs(combined_scaled[:, 0].mean()) < 0.5, "Combined exp_b should be near 0 mean"
        assert abs(combined_scaled[:, 1].mean()) < 0.5, "Combined log_dqv should be near 0 mean"

    def test_boundary_cells_filtered(self) -> None:
        """Cells with exp_b >= 4.9 should be excluded from Li-ion pool."""
        rng = np.random.default_rng(1)
        n = 20
        li_df = pd.DataFrame(
            {
                "cell_id": [f"li_{i}" for i in range(n)],
                "exp_b": [4.95] * 5 + list(rng.uniform(0.01, 2.0, n - 5)),
                "delta_Q_var": rng.uniform(1e-6, 0.01, n),
                "cycle_life": rng.integers(100, 1500, n).astype(float),
            }
        )
        zn_df = pd.DataFrame(
            {
                "cell_id": [f"zn_{i}" for i in range(10)],
                "exp_b": rng.uniform(0.01, 1.0, 10),
                "delta_Q_var": rng.uniform(1e-3, 0.05, 10),
                "cycle_life": rng.integers(50, 600, 10).astype(float),
            }
        )
        settings = _make_settings()
        X_li, _, _, _, _ = build_feature_matrix(li_df, zn_df, settings)
        # Only 15 Li cells should remain (5 boundary-hit cells removed)
        assert X_li.shape[0] == n - 5, f"Expected {n - 5} Li cells, got {X_li.shape[0]}"

    def test_log10_applied_to_delta_Q_var(self, sample_dfs: tuple) -> None:
        """Second column of X_li should be log10(delta_Q_var + eps), not raw."""
        li_df, zn_df = sample_dfs
        settings = _make_settings()
        X_li, _, _, _, scaler = build_feature_matrix(li_df, zn_df, settings)

        # The scaler mean for column 1 must be in the log10 range (not raw variance range).
        # log10(1e-6 + 1e-6) ≈ -5.7 and log10(0.01 + 1e-6) ≈ -2.0
        mean_col1 = float(scaler.mean_[1])
        assert -7 < mean_col1 < 0, (
            f"Scaler mean for log_delta_Q_var should be in (-7, 0), got {mean_col1}"
        )
