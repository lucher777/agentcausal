"""Reproducible synthetic multi-turn agent benchmarks and bundled real Open LLM slice."""

from __future__ import annotations

from importlib import resources
from typing import Dict, Tuple

import numpy as np

from .types import TrajectoryDataset


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def simulate_multiturn_agent(
    n_trajectories: int = 2000,
    horizon: int = 5,
    state_dim: int = 3,
    discount: float = 1.0,
    confounding_strength: float = 1.2,
    seed: int = 42,
) -> TrajectoryDataset:
    """Simulate a confounded multi-turn LLM agent trajectory dataset with exact ground truth.

    Models an ``H``-turn conversational agent where each turn's state ``S_t`` tracks
    query difficulty, risk intensity, and context coherence. The logging behavior
    policy ``pi_b`` activates a guardrail/tool action ``A_t in {0, 1}`` adaptively
    based on ``S_t`` (creating strong time-varying confounding). The target evaluation
    policy ``pi_new`` and baseline policy ``pi_base`` have exact analytical ground-truth
    values and exact turn-level causal attributions computed via high-precision Monte
    Carlo rollouts stored in ``dataset.metadata``.

    Args:
        n_trajectories: Number of logged trajectories ``n``.
        horizon: Number of conversation turns ``H >= 2``.
        state_dim: Feature dimension per turn ``d >= 2``.
        discount: Discount factor ``gamma`` in ``(0, 1]``.
        confounding_strength: Strength of state-to-action confounding in logging policy.
        seed: Random seed for reproducibility.

    Returns:
        ``TrajectoryDataset`` with ground-truth values in ``metadata``.
    """
    if n_trajectories < 20:
        raise ValueError("n_trajectories must be >= 20.")
    if horizon < 2:
        raise ValueError("horizon must be >= 2.")
    if state_dim < 2:
        raise ValueError("state_dim must be >= 2.")

    rng = np.random.default_rng(seed)

    # Turn-specific true action-effect weights (stronger in early/middle turns)
    turn_alpha = np.linspace(0.45, 0.15, horizon)
    state_w = np.zeros(state_dim, dtype=np.float64)
    state_w[0] = 0.55
    state_w[1] = -0.70
    if state_dim > 2:
        state_w[2] = 0.25

    def _policy_prob(s: np.ndarray, mode: str) -> np.ndarray:
        if mode == "behavior":
            logits = confounding_strength * (0.8 * s[:, 1] - 0.5 * s[:, 0])
            return np.clip(_sigmoid(logits), 0.12, 0.88)
        if mode == "new":
            logits = 0.9 + 0.6 * s[:, 1] + 0.2 * s[:, 0]
            return np.clip(_sigmoid(logits), 0.15, 0.90)
        if mode == "base":
            logits = -0.6 + 0.3 * s[:, 1]
            return np.clip(_sigmoid(logits), 0.10, 0.85)
        raise ValueError(mode)

    def _rollout(
        n: int,
        policy_schedule: Tuple[str, ...],
        gen: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = np.zeros((n, horizon, state_dim), dtype=np.float64)
        actions = np.zeros((n, horizon), dtype=np.int64)
        rewards = np.zeros((n, horizon), dtype=np.float64)
        probs = np.zeros((n, horizon), dtype=np.float64)

        s_curr = gen.normal(0.0, 1.0, size=(n, state_dim))
        for t in range(horizon):
            states[:, t, :] = s_curr
            p1 = _policy_prob(s_curr, policy_schedule[t])
            a_t = (gen.random(n) < p1).astype(np.int64)
            actions[:, t] = a_t
            probs[:, t] = np.where(a_t == 1, p1, 1.0 - p1)

            # Reward depends on state, action, and state-action interaction
            r_mean = (
                1.2
                + s_curr @ state_w
                + turn_alpha[t] * a_t
                + 0.35 * a_t * s_curr[:, 1]
                - 0.15 * (s_curr[:, 0] ** 2)
            )
            rewards[:, t] = r_mean + gen.normal(0.0, 0.35, size=n)

            # State transition influenced by prior state and guardrail action
            s_next = 0.45 * s_curr + gen.normal(0.0, 0.65, size=(n, state_dim))
            s_next[:, 1] -= 0.25 * a_t
            s_next[:, 0] += 0.10 * a_t
            s_curr = s_next

        return states, actions, rewards, probs

    states, actions, rewards, behavior_probs = _rollout(
        n_trajectories, tuple(["behavior"] * horizon), rng
    )
    target_action_probs = np.column_stack(
        [_policy_prob(states[:, t, :], "new") for t in range(horizon)]
    )
    baseline_action_probs = np.column_stack(
        [_policy_prob(states[:, t, :], "base") for t in range(horizon)]
    )
    target_probs = np.where(
        actions == 1, target_action_probs, 1.0 - target_action_probs
    )

    # High-precision Monte Carlo oracle ground truth for V(pi_new), V(pi_base), and turn phi_t
    oracle_rng = np.random.default_rng(seed + 99991)
    n_oracle = 50000
    gammas = np.power(discount, np.arange(horizon, dtype=np.float64))
    hybrid_vals = np.zeros(horizon + 1, dtype=np.float64)
    for k in range(horizon + 1):
        sched = tuple(["new"] * k + ["base"] * (horizon - k))
        _, _, r_o, _ = _rollout(n_oracle, sched, oracle_rng)
        hybrid_vals[k] = float((r_o @ gammas).mean())

    true_v_base = float(hybrid_vals[0])
    true_v_new = float(hybrid_vals[horizon])
    true_turn_effects = np.diff(hybrid_vals)

    return TrajectoryDataset(
        states=states,
        actions=actions,
        rewards=rewards,
        behavior_probs=behavior_probs,
        target_probs=target_probs,
        target_action_probs=target_action_probs,
        baseline_action_probs=baseline_action_probs,
        discount=discount,
        metadata={
            "true_v_new": true_v_new,
            "true_v_base": true_v_base,
            "true_total_lift": true_v_new - true_v_base,
            "true_turn_effects": true_turn_effects,
            "naive_logged_mean": float((rewards @ gammas).mean()),
        },
    )


def simulate_guardrail_mediation(
    n_samples: int = 3000,
    seed: int = 42,
) -> Dict[str, object]:
    """Simulate a confounded guardrail mediation dataset with exact analytical NDE/NIE.

    Models an LLM query pipeline with pre-treatment covariates ``X`` (query complexity,
    ambiguity, toxicity risk), confounded guardrail activation ``A in {0, 1}``,
    binary over-refusal mediator ``M in {0, 1}``, safety outcome ``S``, and task utility
    outcome ``Y``. Computes exact ground-truth ``NDE``, ``NIE``, and ``ATE`` on ``Y``
    analytically from the data-generating process.

    Args:
        n_samples: Number of interaction logs ``n``.
        seed: Random seed.

    Returns:
        Dictionary containing ``contexts``, ``guardrail``, ``mediator``, ``utility``,
        ``safety``, ``risk_scores``, and ground-truth scalars ``true_nde``, ``true_nie``,
        ``true_ate``.
    """
    if n_samples < 50:
        raise ValueError("n_samples must be >= 50.")
    rng = np.random.default_rng(seed)

    x = rng.normal(0.0, 1.0, size=(n_samples, 3))
    # Risk score driven by toxicity risk (x[:, 1]) and ambiguity (x[:, 2])
    risk_scores = _sigmoid(1.3 * x[:, 1] + 0.5 * x[:, 2] - 0.2)

    # Confounded guardrail assignment
    p_a = np.clip(_sigmoid(0.9 * x[:, 1] - 0.5 * x[:, 0] + 0.3 * x[:, 2]), 0.12, 0.88)
    a = (rng.random(n_samples) < p_a).astype(np.int64)

    # Over-refusal mediator M in {0, 1}: guardrail increases refusal probability
    p_m1 = _sigmoid(-1.6 + 1.45 + 0.7 * x[:, 1] + 0.6 * x[:, 2])
    p_m0 = _sigmoid(-1.6 + 0.7 * x[:, 1] + 0.6 * x[:, 2])
    p_m = np.where(a == 1, p_m1, p_m0)
    m = (rng.random(n_samples) < p_m).astype(np.float64)

    # Task utility Y: direct clarity/safety-rewrite benefit (+0.18) minus over-refusal penalty (-0.75)
    beta_direct = 0.18
    beta_med = -0.75
    base_y = 0.78 + 0.20 * x[:, 0] - 0.18 * x[:, 1] - 0.08 * (x[:, 0] ** 2)
    utility = base_y + beta_direct * a + beta_med * m + rng.normal(0.0, 0.25, size=n_samples)

    # Safety outcome S in [0, 1]: guardrail strongly raises safety on high-risk queries
    p_safe = _sigmoid(0.3 - 1.6 * x[:, 1] + 2.1 * a * (0.4 + _sigmoid(x[:, 1])) + 0.4 * m)
    safety = (rng.random(n_samples) < p_safe).astype(np.float64)

    # Exact population NDE, NIE, ATE on utility via large-N expectation over X
    x_pop = np.random.default_rng(seed + 77777).normal(0.0, 1.0, size=(200000, 3))
    pm1_pop = _sigmoid(-1.6 + 1.45 + 0.7 * x_pop[:, 1] + 0.6 * x_pop[:, 2])
    pm0_pop = _sigmoid(-1.6 + 0.7 * x_pop[:, 1] + 0.6 * x_pop[:, 2])
    true_nde = float(beta_direct)
    true_nie = float(beta_med * (pm1_pop - pm0_pop).mean())
    true_ate = true_nde + true_nie

    return {
        "contexts": x,
        "guardrail": a,
        "mediator": m,
        "utility": utility,
        "safety": safety,
        "risk_scores": risk_scores,
        "true_nde": true_nde,
        "true_nie": true_nie,
        "true_ate": true_ate,
    }


def load_openllm_slice() -> Dict[str, np.ndarray]:
    """Load the compact bundled real Open LLM multi-benchmark exposure slice.

    Contains a de-identified 60-model by 1,500-item stratified slice across six
    public benchmarks (MMLU, ARC-Challenge, HellaSwag, TruthfulQA, WinoGrande, GSM8K)
    for out-of-the-box testing and demonstrations of ``ExposureAuditor``.

    Returns:
        Dictionary with keys ``responses`` (float32, shape ``(60, 1500)``),
        ``benchmark_ids`` (int32, shape ``(1500,)``), ``model_names`` (str array,
        shape ``(60,)``), and ``benchmark_names`` (str array, shape ``(6,)``).
    """
    data_ref = resources.files("agentcausal.data").joinpath("openllm_exposure_slice.npz")
    with resources.as_file(data_ref) as path:
        npz = np.load(path, allow_pickle=False)
        return {
            "responses": npz["responses"].astype(np.float64),
            "benchmark_ids": npz["benchmark_ids"].astype(np.int64),
            "model_names": npz["model_names"],
            "benchmark_names": npz["benchmark_names"],
        }
