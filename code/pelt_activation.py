"""
PELT-based activation period detector for Zn-ion batteries.
Detects N_act: the cycle where monotonic degradation begins (after activation peak).
"""
import os
import random
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt

from utils import monotone_fraction


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_activation_end(
    capacity_curve: np.ndarray,
    min_activation_cycles: int = 3,
    min_degradation_cycles: int = 10,
    pelt_penalty: float = 2.0,
    pelt_model: str = 'l2'
) -> Tuple[int, dict]:
    """
    Detect the end of the activation period in a Zn-ion capacity curve.

    The activation period is the initial phase where capacity rises (or is
    noisy) before settling into monotonic degradation.

    Parameters
    ----------
    capacity_curve : array of discharge capacity values, one per cycle
    min_activation_cycles : minimum cycles to consider as activation (avoids
        declaring cycle 0 or 1 as N_act)
    min_degradation_cycles : minimum cycles of predominantly monotonic decrease
        that must follow N_act for it to be accepted as valid
    pelt_penalty : PELT penalty parameter (higher → fewer changepoints)
    pelt_model : PELT cost model ('l2', 'rbf', 'normal')

    Returns
    -------
    N_act : int, index of activation period end (0-indexed, inclusive).
        Capacity at N_act is Q(N_act), the reference normalization point.
    info : dict with keys
        - 'method': 'pelt' | 'argmax_fallback' | 'trivial'
        - 'changepoints': list of raw PELT changepoints
        - 'selected_cp': the chosen changepoint (== N_act)
        - 'monotone_fraction_post': fraction of post-N_act cycles decreasing
        - 'n_post_cycles': number of cycles after N_act
    """
    n = len(capacity_curve)

    # ---- trivial short-curve guard ----------------------------------------
    if n <= min_activation_cycles + min_degradation_cycles:
        n_act = int(np.argmax(capacity_curve))
        return n_act, {
            'method': 'trivial',
            'changepoints': [],
            'selected_cp': n_act,
            'monotone_fraction_post': float(
                monotone_fraction(capacity_curve[n_act:])
            ),
            'n_post_cycles': n - n_act,
        }

    # ---- PELT changepoint detection ----------------------------------------
    try:
        algo = rpt.Pelt(model=pelt_model, min_size=min_activation_cycles, jump=1)
        algo.fit(capacity_curve.astype(float))
        result = algo.predict(pen=pelt_penalty)
        # ruptures returns endpoints; last element == n (length), drop it
        changepoints = [cp for cp in result if cp < n]
    except Exception:
        changepoints = []

    # ---- Find best changepoint meeting degradation criterion ---------------
    # We want the *first* changepoint after which capacity shows
    # predominantly monotonic decrease for at least min_degradation_cycles.
    # Scan changepoints from left to right; prefer earlier activation ends.
    selected_cp: Optional[int] = None
    best_mf = -1.0

    for cp in sorted(changepoints):
        if cp < min_activation_cycles:
            continue
        post = capacity_curve[cp:]
        if len(post) < min_degradation_cycles:
            continue
        mf = monotone_fraction(post[:min_degradation_cycles + 20])
        # Accept the first cp where at least 70 % of cycles are declining
        if mf >= 0.70:
            selected_cp = cp
            best_mf = mf
            break  # earliest valid changepoint wins

    # ---- Fallback: argmax --------------------------------------------------
    if selected_cp is None:
        selected_cp = int(np.argmax(capacity_curve))
        # Respect min_activation_cycles lower bound
        selected_cp = max(selected_cp, min_activation_cycles - 1)
        method = 'argmax_fallback'
    else:
        method = 'pelt'

    post_act = capacity_curve[selected_cp:]
    mf_post = float(monotone_fraction(post_act)) if len(post_act) > 1 else 1.0

    info = {
        'method': method,
        'changepoints': changepoints,
        'selected_cp': selected_cp,
        'monotone_fraction_post': mf_post,
        'n_post_cycles': len(post_act),
    }
    return selected_cp, info


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_by_activation_end(
    capacity_curve: np.ndarray,
    N_act: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize capacity curve by Q(N_act) and return the post-activation slice.

    Parameters
    ----------
    capacity_curve : full capacity curve (all cycles)
    N_act : index of activation period end (0-indexed)

    Returns
    -------
    post_activation_curve : capacity_curve[N_act:]  (absolute values, Ah/mAh)
    normalized_curve : post_activation_curve / Q(N_act)  (dimensionless SOH)

    Raises
    ------
    ValueError if Q(N_act) == 0 or N_act is out of bounds.
    """
    if N_act < 0 or N_act >= len(capacity_curve):
        raise ValueError(
            f"N_act={N_act} out of bounds for curve of length {len(capacity_curve)}."
        )
    q_act = float(capacity_curve[N_act])
    if q_act == 0.0:
        raise ValueError(f"Q(N_act={N_act}) is zero; cannot normalize.")

    post = capacity_curve[N_act:].copy()
    return post, post / q_act


# ---------------------------------------------------------------------------
# Visual validation
# ---------------------------------------------------------------------------

def validate_pelt_sample(
    cell_ids: list,
    capacity_curves: dict,
    n_act_results: dict,
    sample_frac: float = 0.05,
    output_dir: str = '/Users/melodysong/code/phd/battery_ml/processed/pelt_validation/'
) -> dict:
    """
    Visual validation of PELT results on a random sample of cells.

    For each sampled cell, saves a PNG showing the capacity curve with the
    detected N_act marked.

    Parameters
    ----------
    cell_ids : list of all cell IDs
    capacity_curves : dict mapping cell_id → np.ndarray of capacity values
    n_act_results : dict mapping cell_id → (N_act, info) tuple
    sample_frac : fraction of cells to sample (min 1, max 20)
    output_dir : directory to save validation plots

    Returns
    -------
    summary : dict with keys
        - 'n_sampled': int
        - 'pelt_fraction': fraction of cells where method == 'pelt'
        - 'argmax_fallback_fraction': fraction using argmax fallback
        - 'mean_monotone_fraction_post': mean monotone_fraction_post across sample
        - 'plot_paths': list of saved plot file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    n_sample = max(1, min(20, int(np.ceil(sample_frac * len(cell_ids)))))
    sampled = random.sample(cell_ids, min(n_sample, len(cell_ids)))

    methods = []
    monotone_fracs = []
    plot_paths = []

    for cid in sampled:
        curve = capacity_curves.get(cid)
        result = n_act_results.get(cid)
        if curve is None or result is None:
            continue

        n_act, info = result if isinstance(result, tuple) else (result, {})
        method = info.get('method', 'unknown')
        mf = info.get('monotone_fraction_post', float('nan'))
        methods.append(method)
        if not np.isnan(mf):
            monotone_fracs.append(mf)

        # --- plot ---
        fig, ax = plt.subplots(figsize=(8, 4))
        cycles = np.arange(len(curve))
        ax.plot(cycles, curve, color='steelblue', linewidth=1.2, label='Capacity')
        ax.axvline(n_act, color='crimson', linestyle='--', linewidth=1.5,
                   label=f'N_act={n_act} ({method})')
        if n_act < len(curve):
            ax.scatter([n_act], [curve[n_act]], color='crimson', zorder=5, s=60)
        ax.set_xlabel('Cycle index')
        ax.set_ylabel('Discharge capacity')
        ax.set_title(f'Cell: {cid}  |  N_act={n_act}  |  method={method}')
        ax.legend(fontsize=9)
        fig.tight_layout()

        fname = f"pelt_{str(cid).replace('/', '_')}.png"
        fpath = os.path.join(output_dir, fname)
        fig.savefig(fpath, dpi=100)
        plt.close(fig)
        plot_paths.append(fpath)

    pelt_frac = (
        sum(1 for m in methods if m == 'pelt') / len(methods) if methods else 0.0
    )
    argmax_frac = (
        sum(1 for m in methods if 'argmax' in m) / len(methods) if methods else 0.0
    )

    return {
        'n_sampled': len(sampled),
        'pelt_fraction': round(pelt_frac, 3),
        'argmax_fallback_fraction': round(argmax_frac, 3),
        'mean_monotone_fraction_post': (
            round(float(np.mean(monotone_fracs)), 3) if monotone_fracs else float('nan')
        ),
        'plot_paths': plot_paths,
    }


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

    rng = np.random.default_rng(0)
    print('=== pelt_activation.py unit tests ===')

    # ---- Test 1: clear activation peak followed by monotone decline --------
    # Simulate: 10 cycles rising, then 40 cycles declining
    activation = np.linspace(1.0, 1.3, 10)
    degradation = np.linspace(1.3, 0.7, 40) + rng.normal(0, 0.005, 40)
    curve1 = np.concatenate([activation, degradation])

    n_act1, info1 = detect_activation_end(curve1, min_activation_cycles=3,
                                           min_degradation_cycles=10)
    _assert(5 <= n_act1 <= 15, f'clear peak N_act in [5,15] (got {n_act1})')
    _assert(info1['n_post_cycles'] > 10, 'sufficient post-activation cycles')
    _assert('method' in info1, 'info has method key')

    # ---- Test 2: no clear activation (flat then drop) ----------------------
    flat = np.ones(5)
    drop = np.linspace(1.0, 0.6, 30) + rng.normal(0, 0.003, 30)
    curve2 = np.concatenate([flat, drop])

    n_act2, info2 = detect_activation_end(curve2, min_activation_cycles=3,
                                           min_degradation_cycles=10)
    _assert(0 <= n_act2 < len(curve2), f'N_act in bounds (got {n_act2})')

    # ---- Test 3: very short curve → trivial fallback -----------------------
    short = np.array([1.0, 1.1, 1.05, 0.98])
    n_act3, info3 = detect_activation_end(short, min_activation_cycles=3,
                                           min_degradation_cycles=10)
    _assert(info3['method'] == 'trivial', f"short curve → trivial (got {info3['method']})")

    # ---- Test 4: normalize_by_activation_end -------------------------------
    cap = np.array([1.0, 1.1, 1.2, 1.15, 1.05, 0.95, 0.85])
    post, norm = normalize_by_activation_end(cap, N_act=2)
    _assert(abs(norm[0] - 1.0) < 1e-9, 'norm[0]==1 at N_act')
    _assert(len(post) == len(cap) - 2, f'post length = n - N_act (got {len(post)})')
    _assert(np.allclose(post, cap[2:]), 'post slice correct')

    # ---- Test 5: normalize_by_activation_end edge: N_act = last index ------
    cap_last = np.array([0.9, 1.0])
    post_last, norm_last = normalize_by_activation_end(cap_last, N_act=1)
    _assert(len(post_last) == 1, 'post has 1 element when N_act at last')

    # ---- Test 6: out-of-bounds N_act raises --------------------------------
    try:
        normalize_by_activation_end(cap, N_act=100)
        _assert(False, 'out-of-bounds raises')
    except ValueError:
        _assert(True, 'out-of-bounds raises')

    # ---- Test 7: validate_pelt_sample (smoke test) -------------------------
    cells = [f'cell_{i}' for i in range(10)]
    curves = {c: np.concatenate([np.linspace(1.0, 1.2, 8),
                                  np.linspace(1.2, 0.7, 30) +
                                  rng.normal(0, 0.005, 30)])
              for c in cells}
    n_act_res = {c: detect_activation_end(curves[c]) for c in cells}

    report = validate_pelt_sample(
        cell_ids=cells,
        capacity_curves=curves,
        n_act_results=n_act_res,
        sample_frac=0.5,
        output_dir='/Users/melodysong/code/phd/battery_ml/processed/pelt_validation/',
    )
    _assert('n_sampled' in report, 'report has n_sampled')
    _assert(report['n_sampled'] >= 1, 'at least 1 cell sampled')
    _assert(0.0 <= report['pelt_fraction'] <= 1.0, 'pelt_fraction in [0,1]')
    _assert(len(report['plot_paths']) >= 1, 'at least 1 plot saved')

    print(f'\nResults: {passed} passed, {failed} failed')
