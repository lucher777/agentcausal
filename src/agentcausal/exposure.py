# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kang Liu
"""Model-free audit of hard-item excess accuracy in benchmark response matrices.

Implements GPU- and CPU-accelerated leave-one-model-out peer-matched counterfactual
projection combined with anytime-valid predictable betting e-processes to detect
hard-tail benchmark exposure/memorization shifts across multi-benchmark LLM response
matrices without parametric Item Response Theory (IRT) assumptions.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import numpy as np

from .anytime import AnytimeAuditor, _resolve_device
from .types import ExposureAuditResult

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class ExposureAuditor:
    """Nonparametric leave-one-model-out hard-tail exposure & shift auditor.

    Given an ``(M, J)`` evaluation response matrix ``Y in [0, 1]`` for ``M`` models
    across ``J`` benchmark items:

    1. Partitions items into a hard-tail stratum ``H = {j : p_j <= hard_threshold}``
       and a strictly disjoint reference stratum ``R = {j : p_j > hard_threshold}``.
    2. Constructs peer capability features ``Z`` exclusively from ``R`` (difficulty
       quantile-bin accuracies and optional per-benchmark accuracies on ``R``) so no
       hard-tail outcome ever leaks into covariates.
    3. Computes exact closed-form leave-one-model-out (LOMO) ridge counterfactual
       expectations ``Y_loo[m, j]`` via Allen's PRESS identity, strictly excluding
       model ``m`` from its own counterfactual predictor.
    4. Applies an anytime-valid predictable betting e-process (``AnytimeAuditor``)
       over the bounded hard-tail excess residuals ``D_{m, j} = Y_{m, j} - Y_loo[m, j]``
       in ``[-1, 1]`` to test ``H_{0, m}: E[D_{m, j} | j in H] <= 0`` with Ville's
       inequality Type-I error guarantee.

    Args:
        hard_threshold: Peer pass-rate cutoff defining the hard-tail stratum (default ``0.20``).
        n_difficulty_bins: Number of reference-stratum difficulty quantile bins.
        ridge_alpha: L2 regularization parameter for leave-one-model-out peer projection.
        alpha: Anytime-valid Type-I error significance threshold (default ``0.05``).
        prior_strength: Pseudo-count for the predictable betting e-process.
        max_bet_fraction: Safe betting cap in ``(0, 1)`` for the e-process.
        device: Compute device (``'auto'``, ``'cuda'``, ``'cuda:0'``, or ``'cpu'``).
        random_state: Deterministic seed for exchangeable item-stream ordering.
    """

    def __init__(
        self,
        hard_threshold: float = 0.20,
        n_difficulty_bins: int = 8,
        ridge_alpha: float = 10.0,
        alpha: float = 0.05,
        prior_strength: float = 5.0,
        max_bet_fraction: float = 0.75,
        device: str = "auto",
        random_state: int = 42,
    ) -> None:
        if not (0.01 < hard_threshold < 0.80):
            raise ValueError(
                f"hard_threshold must lie in (0.01, 0.80), got {hard_threshold}."
            )
        if n_difficulty_bins < 2:
            raise ValueError(f"n_difficulty_bins must be >= 2, got {n_difficulty_bins}.")
        if ridge_alpha <= 0.0:
            raise ValueError(f"ridge_alpha must be positive, got {ridge_alpha}.")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
        self.hard_threshold = float(hard_threshold)
        self.n_difficulty_bins = int(n_difficulty_bins)
        self.ridge_alpha = float(ridge_alpha)
        self.alpha = float(alpha)
        self.prior_strength = float(prior_strength)
        self.max_bet_fraction = float(max_bet_fraction)
        self.device = device
        self.random_state = int(random_state)

    def _build_reference_features_numpy(
        self,
        y: np.ndarray,
        p_item: np.ndarray,
        ref_mask: np.ndarray,
        benchmark_ids: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build standardized model capability features strictly from reference items."""
        m = y.shape[0]
        ref_p = p_item[ref_mask]
        qs = np.quantile(ref_p, np.linspace(0.0, 1.0, self.n_difficulty_bins + 1))
        cols = [np.ones((m, 1), dtype=np.float64)]
        for i in range(self.n_difficulty_bins):
            lo, hi = qs[i], qs[i + 1]
            if i == self.n_difficulty_bins - 1:
                mask = ref_mask & (p_item >= lo) & (p_item <= hi)
            else:
                mask = ref_mask & (p_item >= lo) & (p_item < hi)
            if np.any(mask):
                cols.append(y[:, mask].mean(axis=1, keepdims=True))
        if benchmark_ids is not None:
            for b in np.unique(benchmark_ids):
                mask = ref_mask & (benchmark_ids == b)
                if np.any(mask):
                    cols.append(y[:, mask].mean(axis=1, keepdims=True))
        z = np.hstack(cols)
        z_std = z.copy()
        if z.shape[1] > 1:
            means = z[:, 1:].mean(axis=0, keepdims=True)
            stds = z[:, 1:].std(axis=0, ddof=1, keepdims=True) + 1e-8
            z_std[:, 1:] = (z[:, 1:] - means) / stds
        return z_std

    def audit(
        self,
        responses: np.ndarray,
        model_names: Optional[Sequence[str]] = None,
        benchmark_ids: Optional[np.ndarray] = None,
    ) -> ExposureAuditResult:
        """Audit all ``M`` models in a multi-benchmark response matrix.

        Args:
            responses: Response matrix ``Y`` of shape ``(M, J)`` with entries in ``[0, 1]``.
            model_names: Optional list of ``M`` model names.
            benchmark_ids: Optional integer array of shape ``(J,)`` indexing benchmarks.

        Returns:
            ``ExposureAuditResult`` with excess accuracy, SEs, and anytime-valid e-values.
        """
        t0 = time.perf_counter()
        y_np = np.asarray(responses, dtype=np.float64)
        if y_np.ndim != 2 or y_np.shape[0] < 5 or y_np.shape[1] < 20:
            raise ValueError(
                f"responses must be 2D with M >= 5 and J >= 20, got shape {y_np.shape}."
            )
        if np.isnan(y_np).any():
            raise ValueError("responses must not contain NaN values.")
        if y_np.min() < -1e-9 or y_np.max() > 1.0 + 1e-9:
            raise ValueError("responses entries must lie in [0, 1].")
        y_np = np.clip(y_np, 0.0, 1.0)

        m, j = y_np.shape
        names: List[str] = (
            [f"model_{i}" for i in range(m)]
            if model_names is None
            else [str(x) for x in model_names]
        )
        if len(names) != m:
            raise ValueError(f"Expected {m} model_names, got {len(names)}.")

        b_np: Optional[np.ndarray] = None
        if benchmark_ids is not None:
            b_np = np.asarray(benchmark_ids, dtype=np.int64).reshape(-1)
            if b_np.shape[0] != j:
                raise ValueError(
                    f"benchmark_ids length {b_np.shape[0]} does not match J={j}."
                )

        p_item = y_np.mean(axis=0)
        hard_mask = p_item <= self.hard_threshold
        ref_mask = ~hard_mask
        n_hard = int(hard_mask.sum())
        if n_hard < 5 or int(ref_mask.sum()) < 10:
            raise ValueError(
                f"Insufficient items in hard ({n_hard}) or reference ({int(ref_mask.sum())}) "
                f"stratum at hard_threshold={self.hard_threshold}."
            )

        z_std = self._build_reference_features_numpy(y_np, p_item, ref_mask, b_np)
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n_hard)
        resolved = _resolve_device(self.device)

        if resolved.startswith("cuda"):
            assert torch is not None
            dev = torch.device(resolved)
            y_t = torch.tensor(y_np, dtype=torch.float64, device=dev)
            z_t = torch.tensor(z_std, dtype=torch.float64, device=dev)
            reg = self.ridge_alpha * torch.eye(z_t.shape[1], dtype=torch.float64, device=dev)
            reg[0, 0] = 0.0
            h_mat = z_t @ torch.linalg.solve(z_t.T @ z_t + reg, z_t.T)
            h_diag = torch.diagonal(h_mat).unsqueeze(1)

            hard_idx = torch.tensor(np.where(hard_mask)[0], dtype=torch.long, device=dev)
            ref_idx = torch.tensor(np.where(ref_mask)[0], dtype=torch.long, device=dev)
            y_hard_t = y_t[:, hard_idx]
            y_hat_t = h_mat @ y_hard_t
            denom = torch.clamp(1.0 - h_diag, min=1e-6)
            y_loo_t = torch.clamp((y_hat_t - h_diag * y_hard_t) / denom, 0.0, 1.0)
            diff_t = y_hard_t - y_loo_t

            delta_np = diff_t.mean(dim=1).cpu().numpy()
            se_np = (diff_t.std(dim=1, unbiased=True) / np.sqrt(n_hard)).cpu().numpy()
            hard_acc_np = y_hard_t.mean(dim=1).cpu().numpy()
            exp_hard_np = y_loo_t.mean(dim=1).cpu().numpy()
            ref_acc_np = y_t[:, ref_idx].mean(dim=1).cpu().numpy()

            perm_t = torch.tensor(perm, dtype=torch.long, device=dev)
            diff_ordered = diff_t[:, perm_t].cpu().numpy()
            torch.cuda.synchronize()
        else:
            reg = self.ridge_alpha * np.eye(z_std.shape[1], dtype=np.float64)
            reg[0, 0] = 0.0
            h_mat = z_std @ np.linalg.solve(z_std.T @ z_std + reg, z_std.T)
            h_diag = np.diag(h_mat)[:, None]

            y_hard = y_np[:, hard_mask]
            y_hat = h_mat @ y_hard
            denom = np.maximum(1.0 - h_diag, 1e-6)
            y_loo = np.clip((y_hat - h_diag * y_hard) / denom, 0.0, 1.0)
            diff = y_hard - y_loo

            delta_np = diff.mean(axis=1)
            se_np = diff.std(axis=1, ddof=1) / np.sqrt(n_hard)
            hard_acc_np = y_hard.mean(axis=1)
            exp_hard_np = y_loo.mean(axis=1)
            ref_acc_np = y_np[:, ref_mask].mean(axis=1)
            diff_ordered = diff[:, perm]

        auditor = AnytimeAuditor(
            alpha=self.alpha,
            lower=-1.0,
            upper=1.0,
            prior_strength=self.prior_strength,
            max_bet_fraction=self.max_bet_fraction,
        )
        final_log_e, max_log_e, _ = auditor.batch_eprocess(
            matrix=diff_ordered,
            null_mean=0.0,
            alternative="greater",
            device=resolved,
        )

        ln10 = np.log(10.0)
        log10_e = final_log_e / ln10
        log10_max_e = max_log_e / ln10
        anytime_p = np.minimum(1.0, np.exp(-max_log_e))
        flagged = max_log_e >= np.log(1.0 / self.alpha)
        elapsed = time.perf_counter() - t0

        return ExposureAuditResult(
            model_names=names,
            delta=delta_np,
            se=se_np,
            log10_e_values=log10_e,
            log10_max_e_values=log10_max_e,
            anytime_p_values=anytime_p,
            flagged=flagged,
            hard_accuracy=hard_acc_np,
            expected_hard_accuracy=exp_hard_np,
            reference_accuracy=ref_acc_np,
            n_hard_items=n_hard,
            n_total_items=j,
            alpha=self.alpha,
            hard_quantile_threshold=self.hard_threshold,
            device=resolved,
            elapsed_seconds=float(elapsed),
        )
