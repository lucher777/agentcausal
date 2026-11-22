"""Typed result containers and data structures for ``agentcausal``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrajectoryDataset:
    """Multi-turn LLM agent trajectory dataset for sequential off-policy evaluation.

    Attributes:
        states: Context/state feature tensor of shape ``(n, horizon, state_dim)``.
        actions: Binary or discrete guardrail/tool actions of shape ``(n, horizon)``.
        rewards: Per-turn scalar rewards of shape ``(n, horizon)``.
        behavior_probs: Logging policy probabilities ``pi_b(A_t | S_t)`` of shape ``(n, horizon)``,
            or ``None`` if propensities are to be estimated via cross-fitting.
        target_probs: Target evaluation policy probabilities ``pi_e(A_t | S_t)`` of shape
            ``(n, horizon)`` for the observed action ``A_t``, or target action probabilities
            ``pi_e(a=1 | S_t)`` when ``target_action_probs`` is provided.
        target_action_probs: Optional target probability ``P_e(A_t = 1 | S_t)`` of shape
            ``(n, horizon)`` for binary actions, enabling closed-form Q-function expectation.
        baseline_action_probs: Optional baseline policy probability ``P_0(A_t = 1 | S_t)`` of
            shape ``(n, horizon)`` used for turn-level causal attribution against a baseline.
        discount: Per-turn discount factor ``gamma`` in ``(0, 1]`` (default ``1.0``).
        metadata: Optional dictionary holding ground-truth values or dataset descriptors.
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    behavior_probs: Optional[np.ndarray] = None
    target_probs: Optional[np.ndarray] = None
    target_action_probs: Optional[np.ndarray] = None
    baseline_action_probs: Optional[np.ndarray] = None
    discount: float = 1.0
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate array shapes and numeric constraints."""
        self.states = np.asarray(self.states, dtype=np.float64)
        self.actions = np.asarray(self.actions, dtype=np.int64)
        self.rewards = np.asarray(self.rewards, dtype=np.float64)
        if self.states.ndim != 3:
            raise ValueError(
                f"states must be 3D (n, horizon, state_dim), got shape {self.states.shape}."
            )
        n, h, _ = self.states.shape
        if self.actions.shape != (n, h):
            raise ValueError(
                f"actions shape {self.actions.shape} does not match (n, horizon)={(n, h)}."
            )
        if self.rewards.shape != (n, h):
            raise ValueError(
                f"rewards shape {self.rewards.shape} does not match (n, horizon)={(n, h)}."
            )
        if not (0.0 < self.discount <= 1.0):
            raise ValueError(f"discount must lie in (0, 1], got {self.discount}.")
        if self.behavior_probs is not None:
            self.behavior_probs = np.asarray(self.behavior_probs, dtype=np.float64)
            if self.behavior_probs.shape != (n, h):
                raise ValueError(
                    f"behavior_probs shape {self.behavior_probs.shape} != {(n, h)}."
                )
        if self.target_probs is not None:
            self.target_probs = np.asarray(self.target_probs, dtype=np.float64)
            if self.target_probs.shape != (n, h):
                raise ValueError(
                    f"target_probs shape {self.target_probs.shape} != {(n, h)}."
                )
        if self.target_action_probs is not None:
            self.target_action_probs = np.asarray(self.target_action_probs, dtype=np.float64)
            if self.target_action_probs.shape != (n, h):
                raise ValueError(
                    f"target_action_probs shape {self.target_action_probs.shape} != {(n, h)}."
                )
        if self.baseline_action_probs is not None:
            self.baseline_action_probs = np.asarray(
                self.baseline_action_probs, dtype=np.float64
            )
            if self.baseline_action_probs.shape != (n, h):
                raise ValueError(
                    f"baseline_action_probs shape {self.baseline_action_probs.shape} != {(n, h)}."
                )

    @property
    def n_trajectories(self) -> int:
        """Return the number of trajectories ``n``."""
        return int(self.states.shape[0])

    @property
    def horizon(self) -> int:
        """Return the conversation horizon ``H``."""
        return int(self.states.shape[1])

    @property
    def state_dim(self) -> int:
        """Return the per-turn state feature dimension ``d``."""
        return int(self.states.shape[2])


@dataclass
class DROPEResult:
    """Result of sequential off-policy evaluation across multiple estimators.

    Attributes:
        estimate: Primary doubly-robust (DR) policy value estimate.
        se: Martingale influence-function standard error for the DR estimate.
        ci_low: Lower bound of the two-sided ``(1 - alpha)`` confidence interval.
        ci_high: Upper bound of the two-sided ``(1 - alpha)`` confidence interval.
        alpha: Significance level used for confidence intervals.
        method: Primary estimator name (``"dr"``, ``"wdr"``, ``"dm"``, ``"is"``, ``"wis"``).
        estimates: Dictionary mapping each estimator name to its scalar value estimate.
        standard_errors: Dictionary mapping each estimator name to its standard error.
        influence_scores: Per-trajectory influence function scores ``psi_i`` of shape ``(n,)``.
        effective_sample_size: Kish effective sample size of final cumulative importance weights.
        device: Compute device string used for evaluation (e.g., ``"cuda:0"`` or ``"cpu"``).
    """

    estimate: float
    se: float
    ci_low: float
    ci_high: float
    alpha: float
    method: str
    estimates: Dict[str, float]
    standard_errors: Dict[str, float]
    influence_scores: np.ndarray
    effective_sample_size: float
    device: str


@dataclass
class TurnAttributionResult:
    """Exact turn-level causal attribution decomposing total multi-turn policy lift.

    Attributes:
        turn_effects: Array of shape ``(horizon,)`` giving ``phi_t`` for turns ``t = 1..H``.
        turn_ses: Array of shape ``(horizon,)`` giving martingale sandwich SEs per turn.
        turn_ci_low: Lower confidence bounds per turn of shape ``(horizon,)``.
        turn_ci_high: Upper confidence bounds per turn of shape ``(horizon,)``.
        total_lift: Total estimated value lift ``V(pi_new) - V(pi_base)``.
        total_se: Standard error of the total lift.
        v_new: Estimated value under the target policy ``pi_new``.
        v_base: Estimated value under the baseline policy ``pi_base``.
        additivity_error: Absolute discrepancy ``|sum(phi_t) - (v_new - v_base)|``.
        turn_influence_scores: Per-trajectory per-turn influence scores of shape ``(n, horizon)``.
    """

    turn_effects: np.ndarray
    turn_ses: np.ndarray
    turn_ci_low: np.ndarray
    turn_ci_high: np.ndarray
    total_lift: float
    total_se: float
    v_new: float
    v_base: float
    additivity_error: float
    turn_influence_scores: np.ndarray


@dataclass
class GuardrailMediationResult:
    """Multiply-robust path-specific causal decomposition of an LLM guardrail.

    Attributes:
        ate: Total causal effect ``E[Y(1) - Y(0)]``.
        nde: Natural direct safety/utility effect ``E[Y(1, M(0)) - Y(0, M(0))]``.
        nie: Natural indirect over-refusal/mediation effect ``E[Y(1, M(1)) - Y(1, M(0))]``.
        ate_se: Efficient influence function standard error for ATE.
        nde_se: Efficient influence function standard error for NDE.
        nie_se: Efficient influence function standard error for NIE.
        ate_ci: Two-sided ``(1 - alpha)`` confidence interval for ATE.
        nde_ci: Two-sided ``(1 - alpha)`` confidence interval for NDE.
        nie_ci: Two-sided ``(1 - alpha)`` confidence interval for NIE.
        additivity_error: Exact numerical identity check ``|(nde + nie) - ate|``.
        eif_ate: Per-sample efficient influence scores for ATE of shape ``(n,)``.
        eif_nde: Per-sample efficient influence scores for NDE of shape ``(n,)``.
        eif_nie: Per-sample efficient influence scores for NIE of shape ``(n,)``.
        alpha: Significance level.
    """

    ate: float
    nde: float
    nie: float
    ate_se: float
    nde_se: float
    nie_se: float
    ate_ci: Tuple[float, float]
    nde_ci: Tuple[float, float]
    nie_ci: Tuple[float, float]
    additivity_error: float
    eif_ate: np.ndarray
    eif_nde: np.ndarray
    eif_nie: np.ndarray
    alpha: float


@dataclass
class ParetoPoint:
    """Single point on the counterfactual safety-utility guardrail Pareto frontier.

    Attributes:
        threshold: Policy trigger threshold ``tau``.
        safety_rate: Doubly-robust counterfactual safety rate under threshold ``tau``.
        safety_se: Standard error of the counterfactual safety rate.
        utility_rate: Doubly-robust counterfactual task utility under threshold ``tau``.
        utility_se: Standard error of the counterfactual utility rate.
        trigger_rate: Fraction of queries triggering the guardrail under threshold ``tau``.
        is_pareto_optimal: Whether the point is non-dominated on the safety-utility frontier.
    """

    threshold: float
    safety_rate: float
    safety_se: float
    utility_rate: float
    utility_se: float
    trigger_rate: float
    is_pareto_optimal: bool = False


@dataclass
class AnytimeAuditResult:
    """Anytime-valid sequential betting e-process and confidence sequence summary.

    Attributes:
        e_value: Final sequential betting e-value ``E_n``.
        max_e_value: Running supremum ``sup_{t<=n} E_t`` (Ville's inequality test statistic).
        log10_e_value: Base-10 logarithm of ``E_n``.
        log10_max_e_value: Base-10 logarithm of ``sup_{t<=n} E_t``.
        anytime_p_value: Anytime-valid p-value upper bound ``min(1.0, 1.0 / max_e_value)``.
        rejected: Whether ``max_e_value >= 1 / alpha``.
        stopping_time: First index (1-based) where ``E_t >= 1 / alpha``, or ``None``.
        sample_mean: Final empirical mean of the input stream.
        cs_low: Time-uniform lower confidence sequence trajectory of shape ``(n,)``.
        cs_high: Time-uniform upper confidence sequence trajectory of shape ``(n,)``.
        log_e_trajectory: Natural-log wealth trajectory ``log E_t`` of shape ``(n,)``.
        null_mean: Tested null boundary ``mu_0``.
        alpha: Nominal anytime-valid Type-I error bound.
        alternative: Hypothesis direction (``"greater"``, ``"less"``, or ``"two-sided"``).
    """

    e_value: float
    max_e_value: float
    log10_e_value: float
    log10_max_e_value: float
    anytime_p_value: float
    rejected: bool
    stopping_time: Optional[int]
    sample_mean: float
    cs_low: np.ndarray
    cs_high: np.ndarray
    log_e_trajectory: np.ndarray
    null_mean: float
    alpha: float
    alternative: str


@dataclass
class ExposureAuditResult:
    """Nonparametric doubly-robust benchmark exposure and contamination audit summary.

    Attributes:
        model_names: List of evaluated model identifiers of length ``M``.
        delta: Hard-tail covariate-adjusted excess accuracy ``Delta_m`` of shape ``(M,)``.
        se: Standard error of ``Delta_m`` across hard-tail items of shape ``(M,)``.
        log10_e_values: Anytime-valid betting ``log10(E_m)`` of shape ``(M,)``.
        log10_max_e_values: Running supremum ``log10(sup_t E_{m,t})`` of shape ``(M,)``.
        anytime_p_values: Ville anytime-valid p-value bounds ``min(1, 1 / max E_m)`` of shape ``(M,)``.
        flagged: Boolean array of shape ``(M,)`` where ``max E_m >= 1 / alpha``.
        hard_accuracy: Raw accuracy on hard-tail items per model of shape ``(M,)``.
        expected_hard_accuracy: Leave-one-model-out peer-matched counterfactual accuracy on
            hard-tail items of shape ``(M,)``.
        reference_accuracy: Accuracy on reference (non-hard) items per model of shape ``(M,)``.
        n_hard_items: Number of items in the hard-tail stratum.
        n_total_items: Total number of items audited.
        alpha: Anytime-valid significance threshold.
        hard_quantile_threshold: Peer pass-rate threshold defining the hard-tail stratum.
        device: Compute device string used (``"cuda:0"`` or ``"cpu"``).
        elapsed_seconds: Wall-clock runtime in seconds.
    """

    model_names: List[str]
    delta: np.ndarray
    se: np.ndarray
    log10_e_values: np.ndarray
    log10_max_e_values: np.ndarray
    anytime_p_values: np.ndarray
    flagged: np.ndarray
    hard_accuracy: np.ndarray
    expected_hard_accuracy: np.ndarray
    reference_accuracy: np.ndarray
    n_hard_items: int
    n_total_items: int
    alpha: float
    hard_quantile_threshold: float
    device: str
    elapsed_seconds: float

    def top_flagged(self, k: int = 10) -> List[Dict[str, object]]:
        """Return the top-``k`` models ranked by ``log10_e_values`` descending.

        Args:
            k: Maximum number of ranked models to return.

        Returns:
            List of dictionaries summarizing each top-ranked model.
        """
        order = np.argsort(self.log10_e_values)[::-1][:k]
        rows: List[Dict[str, object]] = []
        for rank, idx in enumerate(order, start=1):
            i = int(idx)
            rows.append(
                {
                    "rank": rank,
                    "model": self.model_names[i],
                    "delta": float(self.delta[i]),
                    "se": float(self.se[i]),
                    "log10_e": float(self.log10_e_values[i]),
                    "log10_max_e": float(self.log10_max_e_values[i]),
                    "anytime_p": float(self.anytime_p_values[i]),
                    "hard_acc": float(self.hard_accuracy[i]),
                    "expected_hard_acc": float(self.expected_hard_accuracy[i]),
                    "ref_acc": float(self.reference_accuracy[i]),
                    "flagged": bool(self.flagged[i]),
                }
            )
        return rows


@dataclass
class EquivalenceReport:
    """Numerical parity verification report across analytical, CPU, and GPU routines.

    Attributes:
        passed: True if every check is within ``tolerance``.
        tolerance: Maximum permitted absolute discrepancy.
        checks: Dictionary mapping check names to observed maximum absolute differences.
        device_checked: Whether CUDA GPU parity was verified on an available GPU.
        device_name: Name of the GPU device if available, else ``"cpu"``.
    """

    passed: bool
    tolerance: float
    checks: Dict[str, float]
    device_checked: bool
    device_name: str
