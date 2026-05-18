"""PELT-based activation-period detector for Zn-ion batteries.

Refactored from ``code/pelt_activation.py`` into a purely functional module
with no global state and all I/O injected as parameters.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import ruptures as rpt

from utils.signals import monotone_fraction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_activation_end(
    capacity_curve: np.ndarray,
    min_activation_cycles: int = 3,
    min_degradation_cycles: int = 10,
    pelt_penalty: float = 2.0,
    pelt_model: str = "l2",
) -> tuple[int, dict]:
    """Detect the end of the activation period in a Zn-ion capacity curve.

    The activation period is the initial phase where capacity rises (or is
    noisy) before settling into monotonic degradation.  Detection uses the
    PELT changepoint algorithm with an argmax fallback.

    Args:
        capacity_curve: Per-cycle discharge capacity values (one per cycle).
        min_activation_cycles: Minimum cycles before a changepoint is valid.
        min_degradation_cycles: Minimum post-activation declining cycles needed
            to accept a changepoint as the activation end.
        pelt_penalty: PELT penalty (higher → fewer changepoints).
        pelt_model: PELT cost model (``'l2'``, ``'rbf'``, or ``'normal'``).

    Returns:
        Tuple of ``(N_act, info)`` where:

        - ``N_act`` is the 0-indexed cycle at which activation ends.
        - ``info`` is a dict with keys ``method``, ``changepoints``,
          ``selected_cp``, ``monotone_fraction_post``, ``n_post_cycles``.
    """
    n = len(capacity_curve)

    # Trivial guard for very short curves
    if n <= min_activation_cycles + min_degradation_cycles:
        n_act = int(np.argmax(capacity_curve))
        return n_act, {
            "method": "trivial",
            "changepoints": [],
            "selected_cp": n_act,
            "monotone_fraction_post": float(monotone_fraction(capacity_curve[n_act:])),
            "n_post_cycles": n - n_act,
        }

    # PELT changepoint detection
    changepoints: list[int] = []
    try:
        algo = rpt.Pelt(model=pelt_model, min_size=min_activation_cycles, jump=1)
        algo.fit(capacity_curve.astype(float))
        result = algo.predict(pen=pelt_penalty)
        changepoints = [cp for cp in result if cp < n]
    except Exception as exc:  # noqa: BLE001
        logger.warning("PELT failed, will use argmax fallback", extra={"error": str(exc)})

    # Find the earliest changepoint meeting the degradation criterion
    selected_cp: Optional[int] = None
    for cp in sorted(changepoints):
        if cp < min_activation_cycles:
            continue
        post = capacity_curve[cp:]
        if len(post) < min_degradation_cycles:
            continue
        mf = monotone_fraction(post[: min_degradation_cycles + 20])
        if mf >= 0.70:
            selected_cp = cp
            break

    # Argmax fallback
    if selected_cp is None:
        selected_cp = max(int(np.argmax(capacity_curve)), min_activation_cycles - 1)
        method = "argmax_fallback"
    else:
        method = "pelt"

    post_act = capacity_curve[selected_cp:]
    mf_post = float(monotone_fraction(post_act)) if len(post_act) > 1 else 1.0

    return selected_cp, {
        "method": method,
        "changepoints": changepoints,
        "selected_cp": selected_cp,
        "monotone_fraction_post": mf_post,
        "n_post_cycles": len(post_act),
    }


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_by_activation_end(
    capacity_curve: np.ndarray,
    n_act: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice and normalise a capacity curve by the activation-end capacity.

    Args:
        capacity_curve: Full per-cycle capacity array.
        n_act: 0-indexed activation-end cycle.

    Returns:
        ``(post_curve, soh_curve)`` where ``post_curve`` is the slice
        ``capacity_curve[n_act:]`` and ``soh_curve`` is that slice divided
        by ``Q(n_act)``.

    Raises:
        ValueError: If ``n_act`` is out of bounds or ``Q(n_act) == 0``.
    """
    if n_act < 0 or n_act >= len(capacity_curve):
        raise ValueError(
            f"n_act={n_act} out of bounds for curve of length {len(capacity_curve)}"
        )
    q_act = float(capacity_curve[n_act])
    if q_act == 0.0:
        raise ValueError(f"Q(n_act={n_act}) is zero; cannot normalise")
    post = capacity_curve[n_act:].copy()
    return post, post / q_act
