# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kang Liu
"""Sequential doubly-robust off-policy evaluation and turn-level causal attribution.

Implements cross-fitted multi-turn Doubly Robust (DR), Weighted Doubly Robust (WDR),
Direct Method (DM), Importance Sampling (IS), and Weighted Importance Sampling (WIS)
estimators for multi-turn LLM agent trajectories (Jiang & Li, 2016; Thomas & Brunskill,
2016; Kallus & Uehara, 2020), together with exact telescoping turn-level causal
attribution and PyTorch CUDA batched evaluation.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

from .anytime import _resolve_device
from .types import DROPEResult, TrajectoryDataset, TurnAttributionResult

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _clip_probs(p: np.ndarray, eps: float) -> np.ndarray:
    """Clip probability arrays strictly away from 0 and 1."""
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def _design_features(states: np.ndarray, actions: Optional[np.ndarray] = None) -> np.ndarray:
    """Construct rich quadratic/interaction features for linear Q-function estimation."""
    s = np.asarray(states, dtype=np.float64)
    if actions is None:
        return np.hstack([s, s * s])
    a = np.asarray(actions, dtype=np.float64).reshape(-1, 1)
    return np.hstack([s, s * s, a, a * s, a * (s * s)])


class SequentialDROPE:
    """Cross-fitted sequential Doubly Robust off-policy evaluator for multi-turn agents.

    Estimates the expected cumulative discounted reward ``V(pi_e)`` of a target
    guardrail/tool-use policy ``pi_e`` from logged trajectories generated under a
    state-confounded behavior policy ``pi_b``. Provides double robustness (consistency
    whenever either the behavior propensities or the backward-inducted Q-functions are
    correctly specified), martingale influence-function standard errors, and exact
    turn-level causal attribution decomposing ``V(pi_new) - V(pi_base)`` across turns.

    Args:
        n_folds: Number of cross-fitting folds ``K >= 2``.
        propensity_clip: Lower/upper clipping margin for estimated propensities.
        ridge_alpha: L2 regularization strength for backward-inducted Q-function models.
        alpha: Significance level for confidence intervals (default ``0.05``).
        device: Compute device (``'auto'``, ``'cuda'``, ``'cuda:0'``, or ``'cpu'``).
        random_state: Random seed for cross-fitting fold assignment.
        propensity_override: Optional callable ``f(states_t, actions_t) -> p(a=1|s)``
            for ablation or misspecification experiments.
        q_override: Optional callable ``f(t, states_t, actions_t) -> Q_t(s, a)``
            for ablation or misspecification experiments.
    """

    def __init__(
        self,
        n_folds: int = 5,
        propensity_clip: float = 0.02,
        ridge_alpha: float = 1.0,
        alpha: float = 0.05,
        device: str = "auto",
        random_state: int = 42,
        propensity_override: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        q_override: Optional[Callable[[int, np.ndarray, np.ndarray], np.ndarray]] = None,
    ) -> None:
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}.")
        if not (0.0 < propensity_clip < 0.25):
            raise ValueError(
                f"propensity_clip must lie in (0, 0.25), got {propensity_clip}."
            )
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
        self.n_folds = int(n_folds)
        self.propensity_clip = float(propensity_clip)
        self.ridge_alpha = float(ridge_alpha)
        self.alpha = float(alpha)
        self.device = device
        self.random_state = int(random_state)
        self.propensity_override = propensity_override
        self.q_override = q_override

    def _fit_nuisances_crossfit(
        self,
        dataset: TrajectoryDataset,
        target_action_probs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cross-fit step-wise propensities ``pi_b(A_t|S_t)``, ``Q_t(S_t, A_t)``, ``V_t(S_t)``.

        Args:
            dataset: Multi-turn ``TrajectoryDataset``.
            target_action_probs: Target probability ``P_e(A_t = 1 | S_t)`` of shape ``(n, H)``.

        Returns:
            Tuple ``(pi_b_obs, q_obs, v_hat)`` each of shape ``(n, H)`` where ``v_hat[:, t]``
            is the expected value ``E_{a ~ pi_e}[Q_t(S_t, a)]``.
        """
        n, h = dataset.n_trajectories, dataset.horizon
        gamma = dataset.discount
        pi_b_obs = np.empty((n, h), dtype=np.float64)
        q_obs = np.empty((n, h), dtype=np.float64)
        v_hat = np.empty((n, h), dtype=np.float64)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for train_idx, val_idx in kf.split(np.arange(n)):
            # 1. Behavior propensities pi_b(A_t | S_t)
            for t in range(h):
                if self.propensity_override is not None:
                    p1_val = _clip_probs(
                        self.propensity_override(dataset.states[val_idx, t], t),
                        self.propensity_clip,
                    )
                    a_val = dataset.actions[val_idx, t]
                    pi_b_obs[val_idx, t] = np.where(a_val == 1, p1_val, 1.0 - p1_val)
                elif dataset.behavior_probs is not None:
                    pi_b_obs[val_idx, t] = _clip_probs(
                        dataset.behavior_probs[val_idx, t], self.propensity_clip
                    )
                else:
                    s_tr = _design_features(dataset.states[train_idx, t])
                    a_tr = dataset.actions[train_idx, t]
                    if np.unique(a_tr).size < 2:
                        p1_val = np.full(val_idx.size, float(a_tr.mean()), dtype=np.float64)
                    else:
                        clf = LogisticRegression(C=1.0, max_iter=500)
                        clf.fit(s_tr, a_tr)
                        p1_val = clf.predict_proba(
                            _design_features(dataset.states[val_idx, t])
                        )[:, 1]
                    p1_val = _clip_probs(p1_val, self.propensity_clip)
                    a_val = dataset.actions[val_idx, t]
                    pi_b_obs[val_idx, t] = np.where(a_val == 1, p1_val, 1.0 - p1_val)

            # 2. Backward-inducted Fitted Q-Evaluation (FQE) for target policy pi_e
            v_next_tr = np.zeros(train_idx.size, dtype=np.float64)
            for t in range(h - 1, -1, -1):
                s_tr = dataset.states[train_idx, t]
                a_tr = dataset.actions[train_idx, t]
                y_tr = dataset.rewards[train_idx, t] + gamma * v_next_tr

                s_val = dataset.states[val_idx, t]
                a_val = dataset.actions[val_idx, t]
                p1_e_val = target_action_probs[val_idx, t]
                p1_e_tr = target_action_probs[train_idx, t]

                if self.q_override is not None:
                    q1_val = np.asarray(
                        self.q_override(t, s_val, np.ones(val_idx.size, dtype=np.int64)),
                        dtype=np.float64,
                    )
                    q0_val = np.asarray(
                        self.q_override(t, s_val, np.zeros(val_idx.size, dtype=np.int64)),
                        dtype=np.float64,
                    )
                    q1_tr = np.asarray(
                        self.q_override(t, s_tr, np.ones(train_idx.size, dtype=np.int64)),
                        dtype=np.float64,
                    )
                    q0_tr = np.asarray(
                        self.q_override(t, s_tr, np.zeros(train_idx.size, dtype=np.int64)),
                        dtype=np.float64,
                    )
                else:
                    reg = Ridge(alpha=self.ridge_alpha)
                    reg.fit(_design_features(s_tr, a_tr), y_tr)
                    q1_val = reg.predict(
                        _design_features(s_val, np.ones(val_idx.size, dtype=np.int64))
                    )
                    q0_val = reg.predict(
                        _design_features(s_val, np.zeros(val_idx.size, dtype=np.int64))
                    )
                    q1_tr = reg.predict(
                        _design_features(s_tr, np.ones(train_idx.size, dtype=np.int64))
                    )
                    q0_tr = reg.predict(
                        _design_features(s_tr, np.zeros(train_idx.size, dtype=np.int64))
                    )

                q_obs[val_idx, t] = np.where(a_val == 1, q1_val, q0_val)
                v_hat[val_idx, t] = p1_e_val * q1_val + (1.0 - p1_e_val) * q0_val
                v_next_tr = p1_e_tr * q1_tr + (1.0 - p1_e_tr) * q0_tr

        return pi_b_obs, q_obs, v_hat

    @staticmethod
    def compute_dr_scores_vectorized(
        rewards: np.ndarray,
        rho_step: np.ndarray,
        q_obs: np.ndarray,
        v_hat: np.ndarray,
        discount: float = 1.0,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """Compute per-trajectory DR, WDR, DM, IS, and WIS scores (CPU or CUDA).

        Implements the exact closed-form sequential Doubly Robust influence score:
        ``psi_DR = V_0(S_0) + sum_{t=0}^{H-1} gamma^t * rho_{0:t} * (R_t + gamma*V_{t+1} - Q_t)``
        where ``V_H == 0`` and ``rho_{0:t} = prod_{u=0}^t (pi_e(A_u|S_u) / pi_b(A_u|S_u))``.

        Args:
            rewards: Array of shape ``(n, H)``.
            rho_step: Per-step importance ratios ``pi_e / pi_b`` of shape ``(n, H)``.
            q_obs: Estimated ``Q_t(S_t, A_t)`` of shape ``(n, H)``.
            v_hat: Estimated ``V_t(S_t)`` of shape ``(n, H)``.
            discount: Discount factor ``gamma`` in ``(0, 1]``.
            device: Target compute device (``'auto'``, ``'cuda'``, ``'cuda:0'``, or ``'cpu'``).

        WDR and WIS scores are delta-method linearisations of the self-normalised
        ratio estimators: their mean equals the point estimate and their sample
        variance estimates the asymptotic variance.

        Returns:
            Tuple ``(dr_scores, wdr_scores, dm_scores, is_scores, wis_scores, resolved_device)``.
        """
        resolved = _resolve_device(device)
        n, h = rewards.shape
        gammas = np.power(discount, np.arange(h, dtype=np.float64))

        if resolved.startswith("cuda"):
            assert torch is not None
            dev = torch.device(resolved)
            r_t = torch.tensor(rewards, dtype=torch.float64, device=dev)
            rho_t = torch.tensor(rho_step, dtype=torch.float64, device=dev)
            q_t = torch.tensor(q_obs, dtype=torch.float64, device=dev)
            v_t = torch.tensor(v_hat, dtype=torch.float64, device=dev)
            g_t = torch.tensor(gammas, dtype=torch.float64, device=dev).unsqueeze(0)

            cum_rho = torch.cumprod(rho_t, dim=1)
            zeros = torch.zeros((n, 1), dtype=torch.float64, device=dev)
            v_next = torch.cat([v_t[:, 1:], zeros], dim=1)
            td_err = r_t + discount * v_next - q_t

            dr_t = v_t[:, 0] + torch.sum(g_t * cum_rho * td_err, dim=1)
            dm_t = v_t[:, 0]
            is_t = torch.sum(g_t * cum_rho * r_t, dim=1)

            mean_cum_rho = torch.clamp(cum_rho.mean(dim=0, keepdim=True), min=1e-12)
            w_norm = cum_rho / mean_cum_rho
            c_td = (w_norm * td_err).mean(dim=0, keepdim=True)
            c_r = (w_norm * r_t).mean(dim=0, keepdim=True)
            wdr_t = v_t[:, 0] + torch.sum(g_t * (w_norm * (td_err - c_td) + c_td), dim=1)
            wis_t = torch.sum(g_t * (w_norm * (r_t - c_r) + c_r), dim=1)

            return (
                dr_t.cpu().numpy(),
                wdr_t.cpu().numpy(),
                dm_t.cpu().numpy(),
                is_t.cpu().numpy(),
                wis_t.cpu().numpy(),
                resolved,
            )

        cum_rho = np.cumprod(rho_step, axis=1)
        v_next = np.hstack([v_hat[:, 1:], np.zeros((n, 1), dtype=np.float64)])
        td_err = rewards + discount * v_next - q_obs

        dr = v_hat[:, 0] + np.sum(gammas[None, :] * cum_rho * td_err, axis=1)
        dm = v_hat[:, 0].copy()
        is_scores = np.sum(gammas[None, :] * cum_rho * rewards, axis=1)

        mean_cum_rho = np.maximum(cum_rho.mean(axis=0, keepdims=True), 1e-12)
        w_norm = cum_rho / mean_cum_rho
        # Self-normalised estimators are ratios of means. Their per-trajectory scores are
        # linearised (delta method) so that the scores average to the point estimate and
        # their sample variance gives a valid standard error.
        c_td = (w_norm * td_err).mean(axis=0, keepdims=True)
        c_r = (w_norm * rewards).mean(axis=0, keepdims=True)
        wdr = v_hat[:, 0] + np.sum(gammas[None, :] * (w_norm * (td_err - c_td) + c_td), axis=1)
        wis = np.sum(gammas[None, :] * (w_norm * (rewards - c_r) + c_r), axis=1)
        return dr, wdr, dm, is_scores, wis, resolved

    @staticmethod
    def compute_dr_scores_recursive(
        rewards: np.ndarray,
        rho_step: np.ndarray,
        q_obs: np.ndarray,
        v_hat: np.ndarray,
        discount: float = 1.0,
    ) -> np.ndarray:
        """Exact backward recursive Jiang & Li (2016) DR formula for parity testing.

        Computes ``V_DR^{(H)} = 0`` and for ``t = H - 1, ..., 0``:
        ``V_DR^{(t)} = V_hat_t + rho_t * (R_t + gamma * V_DR^{(t+1)} - Q_t)``.

        Args:
            rewards: Per-turn rewards of shape ``(n, H)``.
            rho_step: Per-step importance ratios of shape ``(n, H)``.
            q_obs: Estimated Q-values of shape ``(n, H)``.
            v_hat: Estimated state values of shape ``(n, H)``.
            discount: Discount factor ``gamma``.

        Returns:
            Per-trajectory DR scores ``V_DR^{(0)}`` of shape ``(n,)``.
        """
        n, h = rewards.shape
        v_rec = np.zeros(n, dtype=np.float64)
        for t in range(h - 1, -1, -1):
            v_rec = v_hat[:, t] + rho_step[:, t] * (
                rewards[:, t] + discount * v_rec - q_obs[:, t]
            )
        return v_rec

    def evaluate(
        self,
        dataset: TrajectoryDataset,
        target_action_probs: Optional[np.ndarray] = None,
        method: str = "dr",
    ) -> DROPEResult:
        """Evaluate target policy value ``V(pi_e)`` with DR, WDR, DM, IS, and WIS.

        Args:
            dataset: Multi-turn ``TrajectoryDataset``.
            target_action_probs: Optional ``P_e(A_t = 1 | S_t)`` of shape ``(n, H)``.
                Falls back to ``dataset.target_action_probs``.
            method: Primary estimator reported in ``estimate``/``se`` (default ``'dr'``).

        Returns:
            ``DROPEResult`` containing point estimates, martingale SEs, CIs, and influence scores.
        """
        if method not in ("dr", "wdr", "dm", "is", "wis"):
            raise ValueError(f"Unsupported method {method!r}.")
        p1_e = (
            dataset.target_action_probs
            if target_action_probs is None
            else np.asarray(target_action_probs, dtype=np.float64)
        )
        if p1_e is None:
            raise ValueError(
                "target_action_probs must be supplied either on dataset or to evaluate()."
            )
        if p1_e.shape != (dataset.n_trajectories, dataset.horizon):
            raise ValueError(
                f"target_action_probs shape {p1_e.shape} != {(dataset.n_trajectories, dataset.horizon)}."
            )

        pi_b_obs, q_obs, v_hat = self._fit_nuisances_crossfit(dataset, p1_e)
        pi_e_obs = np.where(dataset.actions == 1, p1_e, 1.0 - p1_e)
        rho_step = pi_e_obs / pi_b_obs

        dr, wdr, dm, is_s, wis_s, resolved = self.compute_dr_scores_vectorized(
            rewards=dataset.rewards,
            rho_step=rho_step,
            q_obs=q_obs,
            v_hat=v_hat,
            discount=dataset.discount,
            device=self.device,
        )

        n = dataset.n_trajectories
        score_map: Dict[str, np.ndarray] = {
            "dr": dr,
            "wdr": wdr,
            "dm": dm,
            "is": is_s,
            "wis": wis_s,
        }
        estimates = {k: float(v.mean()) for k, v in score_map.items()}
        ses = {k: float(v.std(ddof=1) / np.sqrt(n)) for k, v in score_map.items()}

        primary_scores = score_map[method]
        est = estimates[method]
        se = ses[method]
        z = float(norm.ppf(1.0 - self.alpha / 2.0))
        cum_rho_final = np.prod(rho_step, axis=1)
        ess = float((cum_rho_final.sum() ** 2) / max(float((cum_rho_final ** 2).sum()), 1e-12))

        return DROPEResult(
            estimate=est,
            se=se,
            ci_low=est - z * se,
            ci_high=est + z * se,
            alpha=self.alpha,
            method=method,
            estimates=estimates,
            standard_errors=ses,
            influence_scores=primary_scores,
            effective_sample_size=ess,
            device=resolved,
        )

    def attribute_turns(
        self,
        dataset: TrajectoryDataset,
        new_action_probs: Optional[np.ndarray] = None,
        base_action_probs: Optional[np.ndarray] = None,
    ) -> TurnAttributionResult:
        """Decompose multi-turn policy lift ``V(pi_new) - V(pi_base)`` by turn ``t = 1..H``.

        Defines the telescoping sequence of hybrid policies ``pi^(k)`` that deploy
        ``pi_new`` on turns ``0, ..., k - 1`` and ``pi_base`` on turns ``k, ..., H - 1``.
        Then turn ``t``'s causal attribution is ``phi_t = V_DR(pi^(t)) - V_DR(pi^(t-1))``
        with per-trajectory influence difference ``psi_i^(t) - psi_i^(t-1)``, guaranteeing
        exact additivity ``sum_{t=1}^H phi_t == V_DR(pi_new) - V_DR(pi_base)`` to machine
        precision alongside paired martingale sandwich standard errors.

        Args:
            dataset: Multi-turn ``TrajectoryDataset``.
            new_action_probs: Target policy ``P_new(A_t = 1 | S_t)`` of shape ``(n, H)``.
            base_action_probs: Baseline policy ``P_base(A_t = 1 | S_t)`` of shape ``(n, H)``.

        Returns:
            ``TurnAttributionResult`` with per-turn attributions, SEs, CIs, and additivity check.
        """
        p_new = (
            dataset.target_action_probs
            if new_action_probs is None
            else np.asarray(new_action_probs, dtype=np.float64)
        )
        p_base = (
            dataset.baseline_action_probs
            if base_action_probs is None
            else np.asarray(base_action_probs, dtype=np.float64)
        )
        if p_new is None or p_base is None:
            raise ValueError(
                "Both new_action_probs and base_action_probs must be provided."
            )

        n, h = dataset.n_trajectories, dataset.horizon
        hybrid_scores = np.empty((h + 1, n), dtype=np.float64)
        for k in range(h + 1):
            p_hyb = p_base.copy()
            if k > 0:
                p_hyb[:, :k] = p_new[:, :k]
            res_k = self.evaluate(dataset, target_action_probs=p_hyb, method="dr")
            hybrid_scores[k] = res_k.influence_scores

        turn_psi = np.diff(hybrid_scores, axis=0).T  # (n, H)
        turn_effects = turn_psi.mean(axis=0)
        turn_ses = turn_psi.std(axis=0, ddof=1) / np.sqrt(n)
        z = float(norm.ppf(1.0 - self.alpha / 2.0))

        total_psi = hybrid_scores[h] - hybrid_scores[0]
        total_lift = float(total_psi.mean())
        total_se = float(total_psi.std(ddof=1) / np.sqrt(n))
        v_base = float(hybrid_scores[0].mean())
        v_new = float(hybrid_scores[h].mean())
        add_err = float(abs(float(turn_effects.sum()) - (v_new - v_base)))

        return TurnAttributionResult(
            turn_effects=turn_effects,
            turn_ses=turn_ses,
            turn_ci_low=turn_effects - z * turn_ses,
            turn_ci_high=turn_effects + z * turn_ses,
            total_lift=total_lift,
            total_se=total_se,
            v_new=v_new,
            v_base=v_base,
            additivity_error=add_err,
            turn_influence_scores=turn_psi,
        )
