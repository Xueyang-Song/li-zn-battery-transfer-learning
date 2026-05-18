"""Unit tests for PELT activation-period detection.

Covers the core ``detect_activation_end`` and ``normalize_by_activation_end``
functions across the main code paths: clean PELT detection, argmax fallback,
trivial short curves, and normalization edge cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.transform.activation import detect_activation_end, normalize_by_activation_end


# ---------------------------------------------------------------------------
# detect_activation_end
# ---------------------------------------------------------------------------

class TestDetectActivationEnd:
    """Tests for the PELT-based activation detector."""

    def _clear_peak_curve(self, rng: np.random.Generator) -> np.ndarray:
        """10 rising + 40 declining cycles."""
        activation = np.linspace(1.0, 1.3, 10)
        degradation = np.linspace(1.3, 0.7, 40) + rng.normal(0, 0.005, 40)
        return np.concatenate([activation, degradation])

    def test_clear_peak_detected_in_range(self) -> None:
        """N_act should be near the top of a clear activation peak."""
        rng = np.random.default_rng(0)
        curve = self._clear_peak_curve(rng)
        n_act, info = detect_activation_end(curve, min_activation_cycles=3, min_degradation_cycles=10)
        assert 5 <= n_act <= 18, f"Expected N_act in [5, 18], got {n_act}"
        assert info["n_post_cycles"] >= 10
        assert "method" in info

    def test_info_keys_present(self) -> None:
        """All required keys must be in the returned info dict."""
        curve = np.linspace(1.2, 0.7, 40)
        _, info = detect_activation_end(curve)
        for key in ("method", "changepoints", "selected_cp", "monotone_fraction_post", "n_post_cycles"):
            assert key in info, f"Missing info key: {key}"

    def test_trivial_short_curve(self) -> None:
        """Very short curves should use the 'trivial' method."""
        short = np.array([1.0, 1.1, 1.05, 0.98])
        n_act, info = detect_activation_end(short, min_activation_cycles=3, min_degradation_cycles=10)
        assert info["method"] == "trivial"
        assert 0 <= n_act < len(short)

    def test_flat_then_drop_fallback(self) -> None:
        """No clear PELT changepoint → argmax fallback."""
        rng = np.random.default_rng(42)
        flat = np.ones(5)
        drop = np.linspace(1.0, 0.6, 30) + rng.normal(0, 0.003, 30)
        curve = np.concatenate([flat, drop])
        n_act, info = detect_activation_end(curve)
        assert 0 <= n_act < len(curve)
        assert info["method"] in ("pelt", "argmax_fallback")

    def test_monotone_decreasing_curve(self) -> None:
        """Pure monotone decay — no activation phase, argmax or PELT at very early cycle."""
        curve = np.linspace(1.0, 0.5, 80)
        n_act, info = detect_activation_end(curve)
        assert n_act >= 0
        assert info["monotone_fraction_post"] >= 0.7

    def test_method_is_string(self) -> None:
        """Method field should always be a non-empty string."""
        curve = np.random.default_rng(5).normal(1.0, 0.05, 50)
        _, info = detect_activation_end(curve)
        assert isinstance(info["method"], str) and len(info["method"]) > 0

    def test_n_act_within_bounds(self) -> None:
        """N_act must be a valid index for any curve."""
        rng = np.random.default_rng(99)
        for _ in range(20):
            n = rng.integers(5, 100)
            curve = rng.uniform(0.5, 1.5, n)
            n_act, _ = detect_activation_end(curve)
            assert 0 <= n_act < n, f"N_act={n_act} out of bounds for curve of length {n}"

    def test_min_activation_cycles_respected(self) -> None:
        """N_act should never be below (min_activation_cycles - 1)."""
        curve = np.array([2.0, 1.0] + list(np.linspace(1.0, 0.5, 30)))
        # Even if argmax is at 0, N_act should be ≥ min_activation_cycles - 1
        min_act = 5
        n_act, _ = detect_activation_end(curve, min_activation_cycles=min_act)
        # The argmax fallback clips to max(argmax, min_act - 1)
        assert n_act >= min_act - 1, f"N_act={n_act} below min_activation_cycles-1={min_act-1}"


# ---------------------------------------------------------------------------
# normalize_by_activation_end
# ---------------------------------------------------------------------------

class TestNormalizeByActivationEnd:
    """Tests for the activation-end normalisation function."""

    def test_normalisation_output_shapes(self) -> None:
        """post and soh arrays must have the expected lengths."""
        curve = np.linspace(1.3, 0.7, 40)
        n_act = 10
        post, soh = normalize_by_activation_end(curve, n_act)
        assert len(post) == len(curve) - n_act
        assert len(soh) == len(post)

    def test_soh_starts_at_one(self) -> None:
        """First element of soh should be 1.0 (Q(n_act)/Q(n_act))."""
        curve = np.linspace(1.3, 0.7, 40)
        n_act = 5
        _, soh = normalize_by_activation_end(curve, n_act)
        assert abs(soh[0] - 1.0) < 1e-9, f"soh[0] should be 1.0, got {soh[0]}"

    def test_out_of_bounds_n_act_raises(self) -> None:
        """n_act out of bounds should raise ValueError."""
        curve = np.ones(10)
        with pytest.raises(ValueError, match="out of bounds"):
            normalize_by_activation_end(curve, n_act=15)

    def test_zero_q_act_raises(self) -> None:
        """Zero Q(n_act) should raise ValueError."""
        curve = np.array([1.0, 0.0, 0.5, 0.4])
        with pytest.raises(ValueError, match="zero"):
            normalize_by_activation_end(curve, n_act=1)

    def test_negative_n_act_raises(self) -> None:
        """Negative n_act should raise ValueError."""
        curve = np.ones(10)
        with pytest.raises(ValueError, match="out of bounds"):
            normalize_by_activation_end(curve, n_act=-1)

    def test_post_curve_is_absolute(self) -> None:
        """post curve should equal the raw capacity slice, not normalised."""
        curve = np.array([1.2, 1.3, 1.1, 0.9, 0.8, 0.7])
        n_act = 2
        post, soh = normalize_by_activation_end(curve, n_act)
        expected_post = curve[n_act:]
        np.testing.assert_allclose(post, expected_post)

    def test_soh_monotonically_decreasing_for_ideal_curve(self) -> None:
        """For a monotonically decreasing curve, soh should also decrease."""
        curve = np.linspace(1.3, 0.5, 50)
        n_act = 5
        _, soh = normalize_by_activation_end(curve, n_act)
        assert np.all(np.diff(soh) <= 1e-9), "soh should be non-increasing for ideal curve"
