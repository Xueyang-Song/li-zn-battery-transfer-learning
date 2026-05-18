"""Battery ML Pipeline CLI entry point.

Usage::

    python main.py --stage preprocess
    python main.py --stage experiment
    python main.py --stage all --job-id myrun01

Stages:
    preprocess  — Load raw data, run feature engineering, save parquet files.
    experiment  — Load canonical feature parquets, run MC experiment, save results.
    calibration — (Placeholder) Model calibration and uncertainty analysis.
    all         — Run preprocess then experiment in sequence.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

# Ensure the battery_ml root is on sys.path when invoked directly.
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import PipelineSettings
from utils.logging_config import get_logger, log_step_end, log_step_start, setup_logging


def _run_preprocess(settings: PipelineSettings, logger) -> None:
    """Preprocess stage: load raw data → compute features → save parquet.

    If canonical feature parquets already exist in ``features/``, this step
    loads them and validates their contents.  Otherwise it falls back to a
    data-absent warning — the raw data loaders require the actual dataset files.
    """
    import numpy as np
    import pandas as pd
    from pipeline.transform.features import build_feature_matrix

    t0 = log_step_start(logger, "preprocess")

    li_path = settings.features_path / "li_ion_scalar_features.parquet"
    zn_path = settings.features_path / "zn_ion_scalar_features.parquet"

    if not li_path.exists() or not zn_path.exists():
        logger.warning(
            "Canonical feature parquets not found — run raw data loaders first",
            extra={
                "li_path": str(li_path),
                "zn_path": str(zn_path),
            },
        )
        log_step_end(logger, "preprocess", start_time=t0, status="skipped")
        return

    li_df = pd.read_parquet(li_path)
    zn_df = pd.read_parquet(zn_path)
    logger.info(
        "Loaded canonical features",
        extra={"n_li": len(li_df), "n_zn": len(zn_df)},
    )

    # Validate feature availability
    for name, df in (("li_ion", li_df), ("zn_ion", zn_df)):
        missing = [c for c in ["exp_b", "delta_Q_var", "cycle_life"] if c not in df.columns]
        if missing:
            logger.error(
                "Missing required columns",
                extra={"dataset": name, "missing_cols": missing},
            )
            raise ValueError(f"Dataset {name!r} is missing columns: {missing}")

    X_li, y_li, X_zn, y_zn, scaler = build_feature_matrix(li_df, zn_df, settings)
    logger.info(
        "Feature matrix built",
        extra={
            "X_li_shape": list(X_li.shape),
            "X_zn_shape": list(X_zn.shape),
            "scaler_mean": scaler.mean_.tolist(),
        },
    )

    # Persist processed feature matrix for the experiment stage
    out_dir = settings.features_path
    out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as _np
    _np.save(str(out_dir / "X_li.npy"), X_li)
    _np.save(str(out_dir / "y_li.npy"), y_li)
    _np.save(str(out_dir / "X_zn.npy"), X_zn)
    _np.save(str(out_dir / "y_zn.npy"), y_zn)
    logger.info("Saved processed feature arrays", extra={"dir": str(out_dir)})

    log_step_end(logger, "preprocess", start_time=t0, n_li=len(y_li), n_zn=len(y_zn))


def _run_experiment(
    settings: PipelineSettings,
    logger,
    job_id: str,
) -> None:
    """Experiment stage: load features → run MC sweep → save results."""
    import numpy as np
    import pandas as pd
    from pipeline.experiment.runner import ExperimentRunner

    t0 = log_step_start(logger, "experiment")

    feat_dir = settings.features_path

    # Load processed arrays (produced by preprocess stage or pre-existing)
    X_li_path = feat_dir / "X_li.npy"
    if not X_li_path.exists():
        logger.error(
            "Processed feature arrays not found — run --stage preprocess first",
            extra={"path": str(X_li_path)},
        )
        sys.exit(1)

    X_li = np.load(str(feat_dir / "X_li.npy"))
    y_li = np.load(str(feat_dir / "y_li.npy"))
    X_zn = np.load(str(feat_dir / "X_zn.npy"))
    y_zn = np.load(str(feat_dir / "y_zn.npy"))

    mu_li = float(np.mean(y_li))
    mu_zn = float(np.mean(y_zn))
    logger.info(
        "Loaded feature arrays",
        extra={
            "n_li": len(y_li),
            "n_zn": len(y_zn),
            "mu_li": round(mu_li, 4),
            "mu_zn": round(mu_zn, 4),
        },
    )

    runner = ExperimentRunner(settings=settings, job_id=job_id)
    results_df = runner.run_monte_carlo(X_li, y_li, X_zn, y_zn, mu_li, mu_zn)

    # Save results
    results_path = settings.results_path
    results_path.mkdir(parents=True, exist_ok=True)
    parquet_path = results_path / "experiment_results.parquet"
    results_df.to_parquet(str(parquet_path), index=False)
    logger.info("Saved experiment results", extra={"path": str(parquet_path)})

    # Summary statistics
    summary = runner.compute_summary(results_df)
    summary_path = results_path / "summary_stats.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Saved summary stats", extra={"path": str(summary_path)})

    log_step_end(
        logger,
        "experiment",
        start_time=t0,
        n_records=len(results_df),
    )


def _run_calibration(settings: PipelineSettings, logger) -> None:
    """Calibration stage (placeholder)."""
    logger.info("Calibration stage is not yet implemented", extra={"step": "calibration"})


def main() -> None:
    """CLI entry point for the battery ML pipeline."""
    parser = argparse.ArgumentParser(
        description="Battery ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["preprocess", "experiment", "calibration", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--job-id",
        default=str(uuid.uuid4())[:8],
        help="Unique run identifier (default: random 8-char UUID)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    import logging
    log_level = getattr(logging, args.log_level)
    setup_logging(job_id=args.job_id, level=log_level)
    logger = get_logger(__name__)

    settings = PipelineSettings()
    logger.info(
        "Pipeline starting",
        extra={
            "stage": args.stage,
            "job_id": args.job_id,
            "base_dir": str(settings.base_dir),
        },
    )

    try:
        if args.stage in ("preprocess", "all"):
            _run_preprocess(settings, logger)

        if args.stage in ("experiment", "all"):
            _run_experiment(settings, logger, job_id=args.job_id)

        if args.stage == "calibration":
            _run_calibration(settings, logger)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.error(
            "Pipeline failed with unhandled exception",
            extra={"error": str(exc), "stage": args.stage},
            exc_info=True,
        )
        sys.exit(1)

    logger.info("Pipeline complete", extra={"job_id": args.job_id, "stage": args.stage})


if __name__ == "__main__":
    main()
