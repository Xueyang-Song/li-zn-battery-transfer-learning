"""Centralised, environment-driven configuration for the battery ML pipeline.

All previously hard-coded paths, thresholds, and hyperparameters live here.
Override any value via environment variables prefixed with ``BATTERY_`` or
by placing a ``.env`` file in the working directory.

Example .env::

    BATTERY_LI_ION_SEVERSON_DIR=data/li_ion/severson
    BATTERY_N_MC_TRIALS=50
"""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    """Pydantic Settings for the battery ML pipeline.

    Attributes:
        base_dir: Root directory of the battery_ml project.
        li_ion_severson_dir: Relative path (from base_dir) to Severson MATR data.
        zn_ion_batches_dir: Relative path (from base_dir) to BatteryLife Zn-ion data.
        features_dir: Directory for computed feature parquet files.
        results_dir: Directory for experiment results and DLQ files.
        figures_dir: Directory for generated figures.
        min_cycle_life: Minimum cycle life to keep a cell in the source pool.
        exp_b_filter_max: Upper bound on exp_b; cells at boundary are dropped.
        qc_valid_threshold: Fraction of peak capacity used for valid-cycle QC.
        sg_window: Savitzky-Golay smoothing window length (must be odd).
        sg_order: Savitzky-Golay polynomial order.
        n_test_cells: Number of Zn-ion cells held out as test set.
        test_set_seed: Random seed for reproducible test-set splitting.
        pelt_penalty: PELT penalty parameter (higher → fewer changepoints).
        pelt_model: PELT cost model ('l2', 'rbf', 'normal').
        min_activation_cycles: Minimum cycles considered as activation phase.
        min_degradation_cycles: Minimum monotone-decline cycles required post N_act.
        n_targets: List of Zn-ion training-set sizes for the N-sweep.
        n_mc_trials: Number of Monte Carlo trials per N_target.
        gp_n_steps: Adam optimisation steps for GP marginal-likelihood training.
        gp_lr: Adam learning rate for GP training.
        exp_b_fit_n_cycles: Cycles from post-activation used to fit exp-b.
        exp_b_filter_eps: Epsilon added before log10 to prevent log(0).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BATTERY_",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Data paths
    # ------------------------------------------------------------------
    base_dir: Path = Path("/Users/melodysong/code/phd/battery_ml")
    li_ion_severson_dir: Path = Path("data/li_ion/severson")
    zn_ion_batches_dir: Path = Path("data/zn_ion/batterylife")
    features_dir: Path = Path("features")
    results_dir: Path = Path("results")
    figures_dir: Path = Path("figures")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    min_cycle_life: int = 5
    exp_b_filter_max: float = 4.9
    qc_valid_threshold: float = 0.95
    sg_window: int = 11
    sg_order: int = 3
    n_test_cells: int = 10
    test_set_seed: int = 42

    # ------------------------------------------------------------------
    # PELT activation detection
    # ------------------------------------------------------------------
    pelt_penalty: float = 2.0
    pelt_model: str = "l2"
    min_activation_cycles: int = 3
    min_degradation_cycles: int = 10

    # ------------------------------------------------------------------
    # Experiment / GP
    # ------------------------------------------------------------------
    n_targets: list[int] = [2, 5, 10, 20, 40]
    n_mc_trials: int = 200
    gp_n_steps: int = 200
    gp_lr: float = 0.05
    exp_b_fit_n_cycles: int = 20
    exp_b_filter_eps: float = 1e-6

    # ------------------------------------------------------------------
    # Derived properties (not env-configurable)
    # ------------------------------------------------------------------
    @property
    def li_ion_severson_path(self) -> Path:
        """Absolute path to the Severson MATR directory."""
        return self.base_dir / self.li_ion_severson_dir

    @property
    def zn_ion_batches_path(self) -> Path:
        """Absolute path to the BatteryLife Zn-ion data directory."""
        return self.base_dir / self.zn_ion_batches_dir

    @property
    def features_path(self) -> Path:
        """Absolute path to the features directory."""
        return self.base_dir / self.features_dir

    @property
    def results_path(self) -> Path:
        """Absolute path to the results directory."""
        return self.base_dir / self.results_dir

    @property
    def figures_path(self) -> Path:
        """Absolute path to the figures directory."""
        return self.base_dir / self.figures_dir

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("sg_window")
    @classmethod
    def sg_window_must_be_odd(cls, v: int) -> int:
        """Ensure Savitzky-Golay window is odd."""
        if v % 2 == 0:
            raise ValueError(f"sg_window must be odd, got {v}")
        return v

    @field_validator("pelt_model")
    @classmethod
    def pelt_model_valid(cls, v: str) -> str:
        """Ensure PELT cost model is one of the supported options."""
        allowed = {"l2", "rbf", "normal"}
        if v not in allowed:
            raise ValueError(f"pelt_model must be one of {allowed}, got {v!r}")
        return v
