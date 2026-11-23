# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kang Liu
"""Tests for GuardrailDecomposer path-specific mediation and Pareto frontier."""

from __future__ import annotations

import numpy as np
import pytest

from agentcausal import GuardrailDecomposer, simulate_guardrail_mediation


def test_guardrail_mediation_recovery_and_exact_additivity():
    """Multiply-robust EIF recovers NDE, NIE, ATE and satisfies NDE+NIE==ATE to <1e-12."""
    data = simulate_guardrail_mediation(n_samples=4000, seed=11)
    res = GuardrailDecomposer(n_folds=5, random_state=11).decompose(
        contexts=data["contexts"],
        guardrail=data["guardrail"],
        mediator=data["mediator"],
        outcome=data["utility"],
    )
    assert res.additivity_error < 1e-12
    assert abs((res.nde + res.nie) - res.ate) < 1e-12
    assert abs(res.nde - data["true_nde"]) < 0.04
    assert abs(res.nie - data["true_nie"]) < 0.04
    assert abs(res.ate - data["true_ate"]) < 0.04
    assert res.nde > 0.0  # Direct clarity/rewrite benefit
    assert res.nie < 0.0  # Indirect over-refusal utility cost


def test_guardrail_mediation_continuous_mediator():
    """Bayes-rule EIF estimator works cleanly for continuous mediators M."""
    rng = np.random.default_rng(29)
    n = 2500
    x = rng.normal(0.0, 1.0, size=(n, 2))
    p_a = 1.0 / (1.0 + np.exp(-(0.5 * x[:, 0] - 0.4 * x[:, 1])))
    a = (rng.random(n) < p_a).astype(np.int64)
    m = 0.6 * a + 0.4 * x[:, 0] + rng.normal(0.0, 0.5, size=n)
    y = 1.0 + 0.30 * a - 0.50 * m + 0.25 * x[:, 1] + rng.normal(0.0, 0.3, size=n)

    res = GuardrailDecomposer(n_folds=5, random_state=29).decompose(x, a, m, y)
    assert res.additivity_error < 1e-12
    assert abs(res.nde - 0.30) < 0.06
    assert abs(res.nie - (-0.30)) < 0.06
    assert abs(res.ate - 0.00) < 0.06


def test_guardrail_pareto_frontier_monotonic_tradeoff():
    """Counterfactual Pareto frontier traces safety vs utility across trigger thresholds."""
    data = simulate_guardrail_mediation(n_samples=2500, seed=37)
    dec = GuardrailDecomposer(n_folds=4, random_state=37)
    taus = np.linspace(0.1, 0.9, 9)
    pts = dec.pareto_frontier(
        contexts=data["contexts"],
        guardrail=data["guardrail"],
        safety_outcome=data["safety"],
        utility_outcome=data["utility"],
        risk_scores=data["risk_scores"],
        thresholds=taus,
    )
    assert len(pts) == len(taus)
    # Trigger rate decreases strictly as threshold rises
    triggers = [p.trigger_rate for p in pts]
    assert triggers[0] > triggers[-1]
    # Lower threshold (more guardrail) gives higher safety than very high threshold
    assert pts[0].safety_rate > pts[-1].safety_rate
    # At least one point is Pareto optimal
    assert any(p.is_pareto_optimal for p in pts)
    assert all(p.safety_se > 0.0 and p.utility_se > 0.0 for p in pts)


def test_guardrail_ci_ordering_and_eif_shapes():
    """EIF score arrays match input size and CIs bracket estimates."""
    data = simulate_guardrail_mediation(n_samples=600, seed=53)
    res = GuardrailDecomposer(n_folds=3, random_state=53).decompose(
        data["contexts"], data["guardrail"], data["mediator"], data["utility"]
    )
    assert res.eif_ate.shape == (600,)
    assert res.eif_nde.shape == (600,)
    assert res.eif_nie.shape == (600,)
    assert res.nde_ci[0] < res.nde < res.nde_ci[1]
    assert res.nie_ci[0] < res.nie < res.nie_ci[1]
    assert res.ate_ci[0] < res.ate < res.ate_ci[1]


def test_guardrail_validation_errors():
    """GuardrailDecomposer validates constructor arguments and binary treatment."""
    with pytest.raises(ValueError):
        GuardrailDecomposer(n_folds=1)
    with pytest.raises(ValueError):
        GuardrailDecomposer(propensity_clip=0.4)
    with pytest.raises(ValueError):
        GuardrailDecomposer(alpha=0.0)
    dec = GuardrailDecomposer(n_folds=3)
    with pytest.raises(ValueError):
        dec.decompose(
            np.zeros((20, 2)),
            np.array([0, 1, 2] * 6 + [0, 1]),
            np.zeros(20),
            np.zeros(20),
        )
    with pytest.raises(ValueError):
        dec.decompose(np.zeros((20, 2)), np.zeros(19, dtype=int), np.zeros(20), np.zeros(20))
