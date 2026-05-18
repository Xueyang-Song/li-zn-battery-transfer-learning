"""Monte Carlo experiment runner for battery lifetime transfer learning.

Implements the N_target × N_MC sweep described in the paper.  Each trial:

1. Randomly samples ``n_target`` Zn-ion cells as the training set.
2. Uses the remaining Zn-ion cells as the validation set.
3. Fits four models: SelectiveMTGP, StandardMTGP, SingleGP, ShuffledMTGP.
4. Records RMSE, NLL, 90%-coverage, and mean predictive std.

Failed trials go to a dead-letter queue (DLQ) and are logged at ERROR level.
The DLQ is saved to ``results/dlq_{job_id}.json`` at the end of the run.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config.settings import PipelineSettings
from pipeline.experiment.models import SelectiveMTGP, ShuffledMTGP, SingleGP, StandardMTGP
from utils.metrics import coverage_90, mean_pred_std, nll_gaussian, rmse_cycles

logger = logging.getLogger(__name__)

_MODEL_NAMES = ["SelectiveMTGP", "StandardMTGP", "SingleGP", "ShuffledMTGP"]


class ExperimentRunner:
    """Orchestrates the N_target × N_MC Monte Carlo experiment.

    Args:
        settings: Pipeline configuration (``n_targets``, ``n_mc_trials``, …).
        job_id: Unique run identifier used for DLQ file naming.
    """

    def __init__(self, settings: PipelineSettings, job_id: str = "default") -> None:
        self._settings = settings
        self._job_id = job_id
        self._failed_cells: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_monte_carlo(
        self,
        X_li: np.ndarray,
        y_li: np.ndarray,
        X_zn: np.ndarray,
        y_zn: np.ndarray,
        mu_li: float,
        mu_zn: float,
    ) -> pd.DataFrame:
        """Run the full N_MC × N_targets Monte Carlo sweep.

        Args:
            X_li: Scaled Li-ion feature matrix, shape ``(n_li, d)``.
            y_li: Log-cycle-life for Li-ion, shape ``(n_li,)``.
            X_zn: Scaled Zn-ion feature matrix, shape ``(n_zn, d)``.
            y_zn: Log-cycle-life for Zn-ion, shape ``(n_zn,)``.
            mu_li: Global mean of Li-ion log-cycle-life (for centring).
            mu_zn: Global mean of Zn-ion log-cycle-life (for centring).

        Returns:
            DataFrame of :class:`~pipeline.validate.schemas.ExperimentTrialResult`
            records with one row per (model, trial, n_target) combination.
        """
        cfg = self._settings
        rng = np.random.default_rng(cfg.test_set_seed)

        y_li_c = y_li - mu_li
        y_zn_c = y_zn - mu_zn

        records: list[dict[str, Any]] = []

        for n_t in cfg.n_targets:
            logger.info(
                "Starting N_target sweep",
                extra={"n_target": n_t, "n_mc": cfg.n_mc_trials},
            )
            for trial in range(cfg.n_mc_trials):
                idx = rng.choice(len(X_zn), size=n_t, replace=False)
                val_mask = np.ones(len(X_zn), dtype=bool)
                val_mask[idx] = False

                X_tr, y_tr = X_zn[idx], y_zn[idx]
                y_tr_c = y_zn_c[idx]
                X_val, y_val = X_zn[val_mask], y_zn[val_mask]

                if len(X_val) == 0:
                    continue

                # 1. SelectiveMTGP
                records.append(
                    self._run_trial(
                        "SelectiveMTGP",
                        n_t,
                        trial,
                        lambda: self._fit_predict_mt(
                            SelectiveMTGP(cfg.gp_n_steps, cfg.gp_lr),
                            X_li, y_li_c, X_tr, y_tr_c, X_val, y_val, mu_zn,
                        ),
                    )
                )

                # 2. StandardMTGP
                records.append(
                    self._run_trial(
                        "StandardMTGP",
                        n_t,
                        trial,
                        lambda: self._fit_predict_mt(
                            StandardMTGP(cfg.gp_n_steps, cfg.gp_lr),
                            X_li, y_li_c, X_tr, y_tr_c, X_val, y_val, mu_zn,
                        ),
                    )
                )

                # 3. SingleGP (GP-Direct)
                records.append(
                    self._run_trial(
                        "SingleGP",
                        n_t,
                        trial,
                        lambda: self._fit_predict_single(
                            SingleGP(cfg.gp_n_steps, cfg.gp_lr),
                            X_tr, y_tr_c, X_val, y_val, mu_zn,
                        ),
                    )
                )

                # 4. ShuffledMTGP
                records.append(
                    self._run_trial(
                        "ShuffledMTGP",
                        n_t,
                        trial,
                        lambda: self._fit_predict_mt(
                            ShuffledMTGP(cfg.gp_n_steps, cfg.gp_lr, rng=rng),
                            X_li, y_li_c, X_tr, y_tr_c, X_val, y_val, mu_zn,
                        ),
                    )
                )

            # Progress log per N_target
            self._log_n_target_summary(records, n_t)

        # Flush DLQ
        self._flush_dlq()

        return pd.DataFrame(records)

    def compute_summary(self, results_df: pd.DataFrame) -> dict[str, Any]:
        """Compute median / Q1 / Q3 / Wilcoxon statistics per (N_target, model).

        Args:
            results_df: Output DataFrame from :meth:`run_monte_carlo`.

        Returns:
            Nested dict: ``{n_target: {model: {median, q1, q3, wilcoxon_p}}}``.
        """
        summary: dict[str, Any] = {}
        for n_t in self._settings.n_targets:
            summary[str(n_t)] = {}
            for model in _MODEL_NAMES:
                vals = (
                    results_df[
                        (results_df["n_target"] == n_t)
                        & (results_df["model_name"] == model)
                    ]["rmse_cycles"]
                    .dropna()
                    .values
                )
                summary[str(n_t)][model] = {
                    "median": float(np.median(vals)) if len(vals) else None,
                    "q1": float(np.percentile(vals, 25)) if len(vals) else None,
                    "q3": float(np.percentile(vals, 75)) if len(vals) else None,
                    "n": int(len(vals)),
                }

            # Wilcoxon: SelectiveMTGP vs SingleGP
            selective = (
                results_df[
                    (results_df["n_target"] == n_t)
                    & (results_df["model_name"] == "SelectiveMTGP")
                ]["rmse_cycles"]
                .dropna()
                .values
            )
            single = (
                results_df[
                    (results_df["n_target"] == n_t)
                    & (results_df["model_name"] == "SingleGP")
                ]["rmse_cycles"]
                .dropna()
                .values
            )
            n_min = min(len(selective), len(single))
            if n_min >= 10:
                try:
                    _, p = wilcoxon(selective[:n_min], single[:n_min])
                    summary[str(n_t)]["wilcoxon_selective_vs_single_p"] = float(p)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Wilcoxon failed for N=%d: %s", n_t, exc)

        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_trial(
        self,
        model_name: str,
        n_target: int,
        trial_id: int,
        fn: "Any",
    ) -> dict[str, Any]:
        """Execute a single trial, catching all exceptions into the DLQ.

        Args:
            model_name: Name of the GP model being evaluated.
            n_target: Number of Zn-ion training cells.
            trial_id: Trial index within the MC sweep.
            fn: Zero-argument callable returning
                ``(rmse, nll, cov90, mean_std)``.

        Returns:
            Dict with keys matching :class:`~pipeline.validate.schemas.ExperimentTrialResult`.
        """
        base = {
            "n_target": n_target,
            "trial_id": trial_id,
            "model_name": model_name,
            "rmse_cycles": float("nan"),
            "nll": float("nan"),
            "coverage_90": float("nan"),
            "mean_pred_std": float("nan"),
        }
        try:
            rmse_v, nll_v, cov90_v, mps_v = fn()
            base.update(
                {
                    "rmse_cycles": rmse_v,
                    "nll": nll_v,
                    "coverage_90": cov90_v,
                    "mean_pred_std": mps_v,
                }
            )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            self._failed_cells.append(
                {
                    "model": model_name,
                    "n_target": n_target,
                    "trial_id": trial_id,
                    "error": str(exc),
                    "traceback": tb,
                    "step": "experiment_trial",
                }
            )
            logger.error(
                "Trial failed — added to DLQ",
                extra={
                    "model": model_name,
                    "n_target": n_target,
                    "trial_id": trial_id,
                    "error": str(exc),
                },
            )
        return base

    @staticmethod
    def _fit_predict_mt(
        model: "StandardMTGP | SelectiveMTGP | ShuffledMTGP",
        X_li: np.ndarray,
        y_li_c: np.ndarray,
        X_tr: np.ndarray,
        y_tr_c: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        mu_zn: float,
    ) -> tuple[float, float, float, float]:
        """Fit an MT-GP and evaluate on the validation set."""
        model.fit(X_li, y_li_c, X_tr, y_tr_c)
        mu, var = model.predict(X_val, mu_zn)
        return (
            rmse_cycles(mu, y_val),
            nll_gaussian(mu, var, y_val),
            coverage_90(mu, var, y_val),
            mean_pred_std(var),
        )

    @staticmethod
    def _fit_predict_single(
        model: SingleGP,
        X_tr: np.ndarray,
        y_tr_c: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        mu_zn: float,
    ) -> tuple[float, float, float, float]:
        """Fit a single-task GP and evaluate on the validation set."""
        model.fit(X_tr, y_tr_c)
        mu_c, var = model.predict(X_val)
        mu = mu_c + mu_zn
        return (
            rmse_cycles(mu, y_val),
            nll_gaussian(mu, var, y_val),
            coverage_90(mu, var, y_val),
            mean_pred_std(var),
        )

    def _log_n_target_summary(
        self, records: list[dict], n_t: int
    ) -> None:
        """Log median RMSE per model for the completed N_target."""
        for model in _MODEL_NAMES:
            vals = [
                r["rmse_cycles"]
                for r in records
                if r["n_target"] == n_t
                and r["model_name"] == model
                and np.isfinite(r["rmse_cycles"])
            ]
            med = float(np.nanmedian(vals)) if vals else float("nan")
            logger.info(
                "N_target summary",
                extra={"n_target": n_t, "model": model, "median_rmse": round(med, 2)},
            )

    def _flush_dlq(self) -> None:
        """Save the dead-letter queue to ``results/dlq_{job_id}.json``."""
        if not self._failed_cells:
            return
        results_path = self._settings.results_path
        results_path.mkdir(parents=True, exist_ok=True)
        dlq_path = results_path / f"dlq_{self._job_id}.json"
        try:
            with open(dlq_path, "w") as fh:
                json.dump(self._failed_cells, fh, indent=2, default=str)
            logger.error(
                "DLQ saved",
                extra={"path": str(dlq_path), "n_failed": len(self._failed_cells)},
            )
        except OSError as exc:
            logger.error("Could not write DLQ: %s", exc)
