# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kang Liu
"""Input-validation and edge-case branches across the public API."""
from __future__ import annotations

import numpy as np
import pytest

import agentcausal.anytime as anytime_mod
from agentcausal import (
    AnytimeAuditor,
    ExposureAuditor,
    GuardrailDecomposer,
    SequentialDROPE,
    TrajectoryDataset,
    simulate_guardrail_mediation,
    simulate_multiturn_agent,
)


# ----------------------------------------------------------------- anytime
def test_resolve_device_variants(monkeypatch):
    assert anytime_mod._resolve_device("auto") in ("cpu", "cuda:0")
    with pytest.raises(ValueError):
        anytime_mod._resolve_device("tpu")
    monkeypatch.setattr(anytime_mod, "torch", None)
    assert anytime_mod._resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError):
        anytime_mod._resolve_device("cuda")


def test_anytime_constructor_and_input_checks():
    with pytest.raises(ValueError):
        AnytimeAuditor(prior_strength=0.0)
    aud = AnytimeAuditor()
    u = np.full(10, 0.5)
    with pytest.raises(ValueError):
        aud._one_sided_log_e_vectorized(u, 0.0, "greater")
    with pytest.raises(ValueError):
        aud._one_sided_log_e_vectorized(u, 0.5, "sideways")
    assert aud._one_sided_log_e_vectorized(np.zeros((3, 0)), 0.5, "greater").shape == (3, 0)
    with pytest.raises(ValueError):
        aud._one_sided_log_e_sequential(np.zeros((2, 2)), 0.5, "greater")
    lo, hi = aud.confidence_sequence(np.array([]))
    assert lo.size == 0 and hi.size == 0


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_batch_eprocess_rejects_bad_input(device):
    aud = AnytimeAuditor()
    with pytest.raises(ValueError):
        aud.batch_eprocess(np.full(5, 0.5), 0.5, device=device)
    with pytest.raises(ValueError):
        aud.batch_eprocess(np.full((2, 5), 0.5), 0.5, alternative="both", device=device)


# ---------------------------------------------------------------- exposure
def _toy_matrix(m=30, j=200, seed=0):
    rng = np.random.default_rng(seed)
    ability = rng.normal(size=(m, 1))
    difficulty = rng.normal(size=(1, j)) + 1.0
    return (rng.random((m, j)) < 1 / (1 + np.exp(-(ability - difficulty)))).astype(float)


def test_exposure_input_checks():
    with pytest.raises(ValueError):
        ExposureAuditor(alpha=1.5)
    y = _toy_matrix()
    aud = ExposureAuditor(device="cpu")
    with pytest.raises(ValueError):
        aud.audit(y, model_names=["a", "b"])
    with pytest.raises(ValueError):
        aud.audit(y, benchmark_ids=np.zeros(3, dtype=int))
    with pytest.raises(ValueError):
        aud.audit(np.ones((30, 200)))  # no hard items at all


# --------------------------------------------------------------- guardrail
def test_guardrail_accepts_one_dimensional_contexts():
    data = simulate_guardrail_mediation(n_samples=1500, seed=5)
    x1 = np.asarray(data["contexts"])[:, 0]
    dec = GuardrailDecomposer(n_folds=3, random_state=5)
    res = dec.decompose(x1, data["guardrail"], data["mediator"], data["utility"])
    assert np.isfinite(res.ate)
    pts = dec.pareto_frontier(x1, data["guardrail"], data["safety"], data["utility"],
                              data["risk_scores"], thresholds=[0.3, 0.6])
    assert len(pts) == 2


# --------------------------------------------------------------------- ope
def test_ope_constant_action_fold_and_missing_probs():
    ds = simulate_multiturn_agent(n_trajectories=600, horizon=3, seed=4, n_oracle=1000)
    actions = ds.actions.copy()
    actions[:, 0] = 1
    ds_const = TrajectoryDataset(ds.states, actions, ds.rewards,
                                 target_action_probs=ds.target_action_probs,
                                 baseline_action_probs=ds.baseline_action_probs)
    est = SequentialDROPE(n_folds=3, device="cpu", random_state=0)
    assert np.isfinite(est.evaluate(ds_const).estimate)

    bare = TrajectoryDataset(ds.states, ds.actions, ds.rewards)
    with pytest.raises(ValueError):
        est.evaluate(bare)
    with pytest.raises(ValueError):
        est.evaluate(bare, target_action_probs=np.full((5, 3), 0.5))
    with pytest.raises(ValueError):
        est.attribute_turns(bare)


def test_expand_features_one_dimensional_and_oracle_guard():
    from agentcausal.guardrail import _expand_features

    assert _expand_features(np.arange(4.0)).shape[0] == 4
    with pytest.raises(ValueError):
        simulate_multiturn_agent(n_trajectories=20, horizon=2, n_oracle=10)


# ------------------------------------------------------------------- types
@pytest.mark.parametrize("field", ["actions", "rewards", "behavior_probs", "target_probs",
                                   "target_action_probs", "baseline_action_probs"])
def test_trajectory_dataset_shape_checks(field):
    n, h, d = 8, 3, 2
    kwargs = dict(states=np.zeros((n, h, d)), actions=np.zeros((n, h)), rewards=np.zeros((n, h)))
    kwargs[field] = np.zeros((n, h + 1))
    with pytest.raises(ValueError):
        TrajectoryDataset(**kwargs)
