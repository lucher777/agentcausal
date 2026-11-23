# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kang Liu
"""Tests for SequentialDROPE off-policy evaluation and turn-level causal attribution."""

from __future__ import annotations

import numpy as np
import pytest

from agentcausal import SequentialDROPE, TrajectoryDataset, simulate_multiturn_agent


def test_drope_recovers_true_policy_value():
    """Cross-fitted DR and WDR accurately recover true V(pi_new) under confounding."""
    traj = simulate_multiturn_agent(n_trajectories=2500, horizon=5, seed=101)
    res = SequentialDROPE(n_folds=5, device="cpu", random_state=101).evaluate(traj)
    true_v = float(traj.metadata["true_v_new"])
    assert abs(res.estimate - true_v) < 3.0 * res.se + 0.04
    assert abs(res.estimates["wdr"] - true_v) < 0.10
    assert res.ci_low < res.estimate < res.ci_high
    assert res.effective_sample_size > 100.0


def test_double_robustness_with_misspecified_q_function():
    """DR remains consistent when Q-function is severely misspecified but propensities are right."""
    traj = simulate_multiturn_agent(n_trajectories=3000, horizon=4, seed=202)
    true_v = float(traj.metadata["true_v_new"])

    # Severely corrupted constant Q-function
    bad_q = lambda t, s, a: np.zeros(s.shape[0], dtype=np.float64)
    res = SequentialDROPE(
        n_folds=4, device="cpu", random_state=202, q_override=bad_q
    ).evaluate(traj)
    assert abs(res.estimates["dm"] - true_v) > 2.0  # DM fails completely
    assert abs(res.estimates["dr"] - true_v) < 0.15  # DR stays consistent


def test_double_robustness_with_misspecified_propensity():
    """DR remains accurate when propensities are corrupted to constant 0.5 but Q is well-fit."""
    traj = simulate_multiturn_agent(n_trajectories=3000, horizon=4, seed=303)
    true_v = float(traj.metadata["true_v_new"])

    bad_prop = lambda s, t: np.full(s.shape[0], 0.5, dtype=np.float64)
    res = SequentialDROPE(
        n_folds=4, device="cpu", random_state=303, propensity_override=bad_prop
    ).evaluate(traj)
    # IS fails badly under corrupted propensity; DR stays close via cross-fitted Q
    assert abs(res.estimates["dr"] - true_v) < abs(res.estimates["is"] - true_v)
    assert abs(res.estimates["dr"] - true_v) < 0.18


def test_drope_estimated_propensities_when_behavior_probs_omitted():
    """SequentialDROPE cross-fits logistic propensities when behavior_probs is None."""
    traj = simulate_multiturn_agent(n_trajectories=1500, horizon=4, seed=404)
    traj_no_prop = TrajectoryDataset(
        states=traj.states,
        actions=traj.actions,
        rewards=traj.rewards,
        behavior_probs=None,
        target_action_probs=traj.target_action_probs,
        baseline_action_probs=traj.baseline_action_probs,
        discount=traj.discount,
    )
    res = SequentialDROPE(n_folds=4, device="cpu", random_state=404).evaluate(traj_no_prop)
    true_v = float(traj.metadata["true_v_new"])
    assert abs(res.estimate - true_v) < 0.18


def test_turn_attribution_exact_additivity_and_recovery():
    """Turn-level attributions sum identically to V(pi_new) - V(pi_base) within 1e-12."""
    traj = simulate_multiturn_agent(n_trajectories=2000, horizon=5, seed=505)
    ope = SequentialDROPE(n_folds=4, device="cpu", random_state=505)
    attr = ope.attribute_turns(traj)
    assert attr.turn_effects.shape == (5,)
    assert attr.additivity_error < 1e-12
    assert abs(float(attr.turn_effects.sum()) - attr.total_lift) < 1e-12
    true_lift = float(traj.metadata["true_total_lift"])
    assert abs(attr.total_lift - true_lift) < 0.18
    # All turn CIs properly ordered
    assert np.all(attr.turn_ci_low < attr.turn_effects)
    assert np.all(attr.turn_effects < attr.turn_ci_high)


def test_vectorized_vs_recursive_dr_exact_parity():
    """Closed-form vectorized DR matches backward recursive Jiang & Li DR to <1e-12."""
    rng = np.random.default_rng(77)
    r = rng.normal(0.5, 1.0, size=(200, 6))
    rho = rng.uniform(0.4, 2.0, size=(200, 6))
    q = rng.normal(1.2, 0.8, size=(200, 6))
    v = rng.normal(1.3, 0.8, size=(200, 6))
    dr_vec, *_ = SequentialDROPE.compute_dr_scores_vectorized(r, rho, q, v, discount=0.9)
    dr_rec = SequentialDROPE.compute_dr_scores_recursive(r, rho, q, v, discount=0.9)
    assert np.max(np.abs(dr_vec - dr_rec)) < 1e-12


def test_drope_all_methods_supported():
    """All five OPE methods ('dr', 'wdr', 'dm', 'is', 'wis') run cleanly."""
    traj = simulate_multiturn_agent(n_trajectories=300, horizon=3, seed=19)
    ope = SequentialDROPE(n_folds=3, device="cpu", random_state=19)
    for m in ("dr", "wdr", "dm", "is", "wis"):
        res = ope.evaluate(traj, method=m)
        assert res.method == m
        assert np.isfinite(res.estimate) and res.se > 0.0


def test_drope_validation_errors():
    """Invalid dataset shapes and estimator arguments raise ValueError."""
    with pytest.raises(ValueError):
        SequentialDROPE(n_folds=1)
    with pytest.raises(ValueError):
        SequentialDROPE(propensity_clip=0.3)
    with pytest.raises(ValueError):
        SequentialDROPE(alpha=1.2)
    with pytest.raises(ValueError):
        TrajectoryDataset(
            states=np.zeros((10, 3)),
            actions=np.zeros((10, 3), dtype=int),
            rewards=np.zeros((10, 3)),
        )
    with pytest.raises(ValueError):
        TrajectoryDataset(
            states=np.zeros((10, 3, 2)),
            actions=np.zeros((10, 4), dtype=int),
            rewards=np.zeros((10, 3)),
        )
    traj = simulate_multiturn_agent(n_trajectories=100, horizon=3, seed=7)
    with pytest.raises(ValueError):
        SequentialDROPE(device="cpu").evaluate(traj, method="invalid")


def test_self_normalised_scores_average_to_ratio_estimates():
    rng = np.random.default_rng(3)
    n, h = 400, 4
    rewards = rng.normal(size=(n, h))
    rho = rng.uniform(0.3, 2.5, size=(n, h))
    q_obs = rng.normal(size=(n, h))
    v_hat = rng.normal(size=(n, h))
    dr, wdr, dm, is_s, wis, _ = SequentialDROPE.compute_dr_scores_vectorized(
        rewards, rho, q_obs, v_hat, discount=0.9, device="cpu"
    )
    w = np.cumprod(rho, axis=1)
    g = 0.9 ** np.arange(h)
    v_next = np.hstack([v_hat[:, 1:], np.zeros((n, 1))])
    td = rewards + 0.9 * v_next - q_obs
    wis_ratio = float(np.sum(g * (w * rewards).mean(0) / w.mean(0)))
    wdr_ratio = float(v_hat[:, 0].mean() + np.sum(g * (w * td).mean(0) / w.mean(0)))
    assert abs(wis.mean() - wis_ratio) < 1e-12
    assert abs(wdr.mean() - wdr_ratio) < 1e-12
    # Constant rewards: the ratio estimator is exactly constant, so its linearised
    # scores must have zero variance (raw normalised products would not).
    const = np.full((n, h), 2.0)
    _, _, _, _, wis_c, _ = SequentialDROPE.compute_dr_scores_vectorized(
        const, rho, q_obs, v_hat, discount=0.9, device="cpu"
    )
    assert np.allclose(wis_c, 2.0 * g.sum(), atol=1e-12)
