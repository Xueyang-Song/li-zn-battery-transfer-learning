"""Pydantic v2 data schemas for the battery ML pipeline.

Every stage of the pipeline — raw extraction, feature engineering, and
experiment results — is validated against one of these schemas.  Validators
enforce domain-specific constraints (e.g. positive cycle life, b ∈ [0, 5]).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RawCellRecord(BaseModel):
    """A single cell's raw data as it arrives from the extraction stage.

    Attributes:
        cell_id: Unique cell identifier (e.g. ``"severson_train_cell1"``).
        chemistry: Battery chemistry string (``"LFP"``, ``"ZnMnO2"``, etc.).
        dataset: Source dataset name (``"severson_matr"``, ``"batterylife"``, …).
        cycle_life: Total cycles to end-of-life (positive integer).
        raw_capacity_curve: Per-cycle discharge capacity values (Ah or mAh).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cell_id: str = Field(..., description="Unique cell identifier")
    chemistry: str = Field(..., description="Battery chemistry")
    dataset: str = Field(..., description="Source dataset name")
    cycle_life: float = Field(..., gt=0, description="Cycle life (positive)")
    raw_capacity_curve: list[float] = Field(
        ..., min_length=3, description="Per-cycle discharge capacity values"
    )

    @field_validator("cell_id")
    @classmethod
    def cell_id_not_empty(cls, v: str) -> str:
        """Reject empty cell IDs."""
        if not v.strip():
            raise ValueError("cell_id must not be empty")
        return v.strip()

    @field_validator("raw_capacity_curve")
    @classmethod
    def curve_must_be_finite(cls, v: list[float]) -> list[float]:
        """Reject curves containing NaN or ±Inf."""
        if any(not np.isfinite(x) for x in v):
            raise ValueError("raw_capacity_curve contains non-finite values")
        return v


class FeatureRecord(BaseModel):
    """Scalar features extracted from a single cell.

    Attributes:
        cell_id: Unique cell identifier.
        chemistry: Battery chemistry string.
        dataset: Source dataset name.
        cycle_life: Total cycles to end-of-life.
        exp_b: Exponential decay rate ``b`` from Q(n)/Q0 = A+(1-A)·exp(−b·n).
        delta_Q_var: Variance of SG-smoothed dQ/dV curves.
        log_delta_Q_var: ``log10(delta_Q_var + 1e-6)`` — compressed scale.
        Q_rel_10: Relative capacity at cycle 10 (Q_10/Q_0).
        n_act: Activation-period end index (Zn-ion only; None for Li-ion).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cell_id: str
    chemistry: str
    dataset: str
    cycle_life: float = Field(..., gt=0)
    exp_b: float = Field(..., description="Exponential decay rate b ∈ [0, 5]")
    delta_Q_var: float = Field(..., ge=0, description="Variance of dQ/dV (non-negative)")
    log_delta_Q_var: float = Field(..., description="log10(delta_Q_var + 1e-6)")
    Q_rel_10: Optional[float] = Field(None, ge=0, description="Q_10/Q_0 (optional)")
    n_act: Optional[int] = Field(None, ge=0, description="Activation end index (Zn only)")

    @field_validator("exp_b")
    @classmethod
    def exp_b_in_range(cls, v: float) -> float:
        """exp_b must be a non-negative finite value below the boundary."""
        if not np.isfinite(v) or v < 0 or v > 5.0:
            raise ValueError(f"exp_b={v} outside valid range [0, 5]")
        return v

    @model_validator(mode="after")
    def log_transform_consistent(self) -> "FeatureRecord":
        """Verify log_delta_Q_var is consistent with delta_Q_var."""
        expected = float(np.log10(self.delta_Q_var + 1e-6))
        if abs(self.log_delta_Q_var - expected) > 1e-4:
            raise ValueError(
                f"log_delta_Q_var={self.log_delta_Q_var:.6f} inconsistent "
                f"with delta_Q_var={self.delta_Q_var:.6e} (expected {expected:.6f})"
            )
        return self


class ExperimentTrialResult(BaseModel):
    """Outcome of a single Monte Carlo trial.

    Attributes:
        n_target: Number of Zn-ion training cells in this trial.
        trial_id: Trial index (0-indexed).
        model_name: Model identifier (e.g. ``"SelectiveMTGP"``).
        rmse_cycles: RMSE in cycle space (non-negative).
        nll: Mean negative log-likelihood (may be negative for good models).
        coverage_90: Empirical 90 % PI coverage ∈ [0, 1].
        mean_pred_std: Mean predictive standard deviation (sharpness).
    """

    n_target: int = Field(..., gt=0)
    trial_id: int = Field(..., ge=0)
    model_name: str
    rmse_cycles: float = Field(..., ge=0)
    nll: float
    coverage_90: float = Field(..., ge=0, le=1)
    mean_pred_std: float = Field(..., ge=0)

    @field_validator("model_name")
    @classmethod
    def model_name_not_empty(cls, v: str) -> str:
        """Reject blank model names."""
        if not v.strip():
            raise ValueError("model_name must not be empty")
        return v.strip()
