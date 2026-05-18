"""GPyTorch GP models for battery lifetime transfer learning.

Four models are implemented:

- :class:`SelectiveMTGP`: Transfers only ``exp_b`` across tasks via a
  product-kernel architecture.  This is the primary model in the paper.
- :class:`StandardMTGP`: ICM-style MT-GP with full feature sharing.
- :class:`SingleGP`: Single-task Matérn-5/2 GP (GP-Direct baseline).
- :class:`ShuffledMTGP`: MT-GP with shuffled Li-ion labels (control).

All models expose a uniform ``fit`` / ``predict`` interface.
"""

from __future__ import annotations

import logging
from typing import Optional

import gpytorch
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPyTorch model definitions
# ---------------------------------------------------------------------------

class _ExactGPModel(gpytorch.models.ExactGP):
    """Single-task Exact GP with Matérn-5/2 kernel and ARD.

    Args:
        train_x: Training inputs tensor, shape ``(n, d)``.
        train_y: Training targets tensor, shape ``(n,)``.
        likelihood: GPyTorch Gaussian likelihood.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[1])
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:  # noqa: D102
        return gpytorch.distributions.MultivariateNormal(self.mean(x), self.covar(x))


class _MTGPModel(gpytorch.models.ExactGP):
    """Multi-task GP via data-kernel × task-IndexKernel (ICM approximation).

    Last column of ``train_x`` must be the integer task index (0=Li, 1=Zn).

    Args:
        train_x: Joint training inputs, shape ``(n_all, d+1)``.
        train_y: Joint training targets, shape ``(n_all,)``.
        likelihood: GPyTorch Gaussian likelihood.
        n_tasks: Number of tasks (default 2).
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        n_tasks: int = 2,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        d_feat = train_x.shape[1] - 1  # exclude task index column
        self.mean = gpytorch.means.ConstantMean()
        self.data_covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d_feat)
        )
        self.task_covar = gpytorch.kernels.IndexKernel(num_tasks=n_tasks, rank=1)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:  # noqa: D102
        x_feat = x[..., :-1]
        x_task = x[..., -1:].long()
        mean = self.mean(x_feat)
        covar = self.data_covar(x_feat).mul(self.task_covar(x_task))
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class _SelectiveMTGPModel(gpytorch.models.ExactGP):
    """Selective MT-GP that transfers only the ``exp_b`` feature.

    Architecture::

        k_shared(exp_b) × k_task(task_idx)
        + k_private(log_dqv) × mask(Zn-only)

    Input layout: ``x = [exp_b_scaled, log_dqv_scaled, task_idx]``

    - Column 0: ``exp_b`` (shared across tasks)
    - Column 1: ``log_delta_Q_var`` (private, Zn-specific)
    - Column 2: task index (0=Li, 1=Zn) as float (cast to long internally)

    Args:
        train_x: Joint training inputs, shape ``(n_all, 3)``.
        train_y: Joint training targets, shape ``(n_all,)``.
        likelihood: GPyTorch Gaussian likelihood.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        # Shared: exp_b × task covariance
        self.k_shared_feat = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=1)
        )
        self.k_task = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)
        # Private: log_dqv (used only for Zn-Zn pairs via soft mask in training data)
        self.k_private = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=1)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:  # noqa: D102
        expb = x[..., :1]           # shape (..., 1)
        log_dqv = x[..., 1:2]      # shape (..., 1)
        task = x[..., 2:3].long()  # shape (..., 1)

        mean = self.mean(expb)
        k_shared = self.k_shared_feat(expb).mul(self.k_task(task))
        k_priv = self.k_private(log_dqv)
        covar = k_shared.add(k_priv)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ---------------------------------------------------------------------------
# Public model classes with fit / predict interface
# ---------------------------------------------------------------------------

def _train_gp(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    n_steps: int,
    lr: float,
) -> None:
    """Train a GP model in-place using Adam + exact MLL."""
    model.train()
    likelihood.train()
    optimiser = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for step in range(n_steps):
        optimiser.zero_grad()
        try:
            loss = -mll(model(X_t), y_t)
            loss.backward()
            optimiser.step()
        except Exception as exc:  # noqa: BLE001
            logger.debug("GP training step %d failed: %s", step, exc)
            break


class SingleGP:
    """Standard single-task Matérn-5/2 GP (GP-Direct baseline).

    Operates purely on Zn-ion data; no Li-ion source data used.

    Args:
        n_steps: Adam optimisation steps (default 200).
        lr: Adam learning rate (default 0.05).
    """

    def __init__(self, n_steps: int = 200, lr: float = 0.05) -> None:
        self.n_steps = n_steps
        self.lr = lr
        self._model: Optional[_ExactGPModel] = None
        self._likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SingleGP":
        """Fit the GP to training data.

        Args:
            X_train: Feature matrix, shape ``(n, d)``.
            y_train: Target vector (log-cycle-life), shape ``(n,)``.

        Returns:
            Self (for method chaining).
        """
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._model = _ExactGPModel(X_t, y_t, self._likelihood)
        _train_gp(self._model, self._likelihood, X_t, y_t, self.n_steps, self.lr)
        return self

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predictive mean and variance.

        Args:
            X_test: Test feature matrix, shape ``(m, d)``.

        Returns:
            Tuple ``(mean, variance)`` each shape ``(m,)`` in log-cycle space.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        self._model.eval()
        self._likelihood.eval()
        X_t = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._model(X_t))
        return pred.mean.numpy(), pred.variance.numpy()


class StandardMTGP:
    """ICM Multi-Task GP with full feature sharing.

    Concatenates Li-ion (task=0) and Zn-ion (task=1) data; the task index is
    appended as an extra column.

    Args:
        n_steps: Adam optimisation steps (default 200).
        lr: Adam learning rate (default 0.05).
    """

    def __init__(self, n_steps: int = 200, lr: float = 0.05) -> None:
        self.n_steps = n_steps
        self.lr = lr
        self._model: Optional[_MTGPModel] = None
        self._likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self._mu_zn: float = 0.0

    def fit(
        self,
        X_li: np.ndarray,
        y_li_centered: np.ndarray,
        X_zn: np.ndarray,
        y_zn_centered: np.ndarray,
    ) -> "StandardMTGP":
        """Fit on concatenated Li+Zn data (pre-centred labels).

        Args:
            X_li: Li-ion feature matrix, shape ``(n_li, d)``.
            y_li_centered: Mean-centred Li-ion log-cycle-life, shape ``(n_li,)``.
            X_zn: Zn-ion feature matrix, shape ``(n_zn, d)``.
            y_zn_centered: Mean-centred Zn-ion log-cycle-life, shape ``(n_zn,)``.

        Returns:
            Self.
        """
        n_li, n_zn = len(X_li), len(X_zn)
        X_all = np.vstack([
            np.hstack([X_li, np.zeros((n_li, 1))]),
            np.hstack([X_zn, np.ones((n_zn, 1))]),
        ])
        y_all = np.concatenate([y_li_centered, y_zn_centered])
        X_t = torch.tensor(X_all, dtype=torch.float32)
        y_t = torch.tensor(y_all, dtype=torch.float32)
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._model = _MTGPModel(X_t, y_t, self._likelihood)
        _train_gp(self._model, self._likelihood, X_t, y_t, self.n_steps, self.lr)
        return self

    def predict(
        self,
        X_test: np.ndarray,
        mu_zn: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict on Zn-ion test cells (task=1) and add back global mean.

        Args:
            X_test: Zn-ion test feature matrix, shape ``(m, d)``.
            mu_zn: Global mean of Zn-ion log-cycle-life (added to centred predictions).

        Returns:
            Tuple ``(mean, variance)`` each shape ``(m,)`` in log-cycle space.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        self._model.eval()
        self._likelihood.eval()
        X_te = np.hstack([X_test, np.ones((len(X_test), 1))])
        X_t = torch.tensor(X_te, dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._model(X_t))
        return pred.mean.numpy() + mu_zn, pred.variance.numpy()


class SelectiveMTGP:
    """Selective MT-GP that transfers only ``exp_b`` across tasks.

    Architecture::

        k_shared(exp_b) × k_task(task_idx)
        + k_private(log_dqv) × I(Zn-Zn)

    Input layout: ``X = [exp_b_scaled, log_dqv_scaled]`` (2 columns).
    Internally appends the task-index column.

    Args:
        n_steps: Adam optimisation steps (default 200).
        lr: Adam learning rate (default 0.05).
    """

    def __init__(self, n_steps: int = 200, lr: float = 0.05) -> None:
        self.n_steps = n_steps
        self.lr = lr
        self._model: Optional[_SelectiveMTGPModel] = None
        self._likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None

    def fit(
        self,
        X_li: np.ndarray,
        y_li_centered: np.ndarray,
        X_zn: np.ndarray,
        y_zn_centered: np.ndarray,
    ) -> "SelectiveMTGP":
        """Fit the selective MT-GP on pre-centred labels.

        Args:
            X_li: Li-ion features ``[exp_b, log_dqv]``, shape ``(n_li, 2)``.
            y_li_centered: Centred Li-ion log-cycle-life.
            X_zn: Zn-ion features ``[exp_b, log_dqv]``, shape ``(n_zn, 2)``.
            y_zn_centered: Centred Zn-ion log-cycle-life.

        Returns:
            Self.
        """
        n_li, n_zn = len(X_li), len(X_zn)
        X_all = np.vstack([
            np.hstack([X_li, np.zeros((n_li, 1))]),
            np.hstack([X_zn, np.ones((n_zn, 1))]),
        ])
        y_all = np.concatenate([y_li_centered, y_zn_centered])
        X_t = torch.tensor(X_all, dtype=torch.float32)
        y_t = torch.tensor(y_all, dtype=torch.float32)
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._model = _SelectiveMTGPModel(X_t, y_t, self._likelihood)
        _train_gp(self._model, self._likelihood, X_t, y_t, self.n_steps, self.lr)
        return self

    def predict(
        self,
        X_test: np.ndarray,
        mu_zn: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict on Zn-ion test cells and add back global mean.

        Args:
            X_test: Zn-ion test features ``[exp_b, log_dqv]``, shape ``(m, 2)``.
            mu_zn: Global mean of Zn-ion log-cycle-life.

        Returns:
            Tuple ``(mean, variance)`` each shape ``(m,)`` in log-cycle space.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        self._model.eval()
        self._likelihood.eval()
        X_te = np.hstack([X_test, np.ones((len(X_test), 1))])
        X_t = torch.tensor(X_te, dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._model(X_t))
        return pred.mean.numpy() + mu_zn, pred.variance.numpy()


class ShuffledMTGP:
    """MT-GP control with shuffled Li-ion labels.

    Identical architecture to :class:`StandardMTGP` but the Li-ion labels are
    randomly permuted before training to break the cross-chemistry signal.
    A consistent ``rng`` ensures reproducible shuffles.

    Args:
        n_steps: Adam optimisation steps (default 200).
        lr: Adam learning rate (default 0.05).
        rng: NumPy random generator for reproducible shuffling.
    """

    def __init__(
        self,
        n_steps: int = 200,
        lr: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.n_steps = n_steps
        self.lr = lr
        self._rng = rng if rng is not None else np.random.default_rng()
        self._delegate = StandardMTGP(n_steps=n_steps, lr=lr)

    def fit(
        self,
        X_li: np.ndarray,
        y_li_centered: np.ndarray,
        X_zn: np.ndarray,
        y_zn_centered: np.ndarray,
    ) -> "ShuffledMTGP":
        """Fit with permuted Li-ion labels.

        Args:
            X_li: Li-ion feature matrix.
            y_li_centered: Centred Li-ion log-cycle-life (will be shuffled).
            X_zn: Zn-ion feature matrix.
            y_zn_centered: Centred Zn-ion log-cycle-life (unchanged).

        Returns:
            Self.
        """
        y_li_shuffled = self._rng.permutation(y_li_centered)
        self._delegate.fit(X_li, y_li_shuffled, X_zn, y_zn_centered)
        return self

    def predict(
        self,
        X_test: np.ndarray,
        mu_zn: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict (delegates to internal StandardMTGP).

        Args:
            X_test: Zn-ion test feature matrix.
            mu_zn: Global mean of Zn-ion log-cycle-life.

        Returns:
            Tuple ``(mean, variance)`` in log-cycle space.
        """
        return self._delegate.predict(X_test, mu_zn)
