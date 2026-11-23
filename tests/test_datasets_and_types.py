# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kang Liu
"""Tests for dataset generators, typed containers, GPU execution, and edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from agentcausal import (
    AnytimeAuditor,
    ExposureAuditor,
    GuardrailDecomposer,
    SequentialDROPE,
    TrajectoryDataset,
    __version__,
    load_openllm_slice,
    simulate_guardrail_mediation,
    simulate_multiturn_agent,
)

try:
    import torch

    HAS_CUDA = bool(torch.cuda.is_available())
except ImportError:
    HAS_CUDA = False


def test_package_version_is_semver():
    """Package exposes semantic version string."""
    assert __version__ == "0.1.0"


def test_trajectory_dataset_properties_and_discount():
    """TrajectoryDataset exposes n_trajectories, horizon, state_dim and validates discount."""
    ds = simulate_multiturn_agent(n_trajectories=120, horizon=4, state_dim=3, discount=0.9, seed=1)
    assert ds.n_trajectories == 120
    assert ds.horizon == 4
    assert ds.state_dim == 3
    assert ds.discount == 0.9
    with pytest.raises(ValueError):
        TrajectoryDataset(ds.states, ds.actions, ds.rewards, discount=0.0)
    with pytest.raises(ValueError):
        TrajectoryDataset(ds.states, ds.actions, ds.rewards, discount=1.1)


@pytest.mark.parametrize("discount", [0.85, 0.95, 1.0])
def test_drope_supports_multiple_discount_factors(discount):
    """SequentialDROPE accurately evaluates trajectories under different gamma values."""
    ds = simulate_multiturn_agent(n_trajectories=800, horizon=4, discount=discount, seed=9)
    res = SequentialDROPE(n_folds=4, device="cpu", random_state=9).evaluate(ds)
    assert abs(res.estimate - float(ds.metadata["true_v_new"])) < 0.20


@pytest.mark.parametrize("horizon", [2, 4, 6])
def test_turn_attribution_scales_across_horizons(horizon):
    """Turn attribution vector length and exact additivity hold across horizons H."""
    ds = simulate_multiturn_agent(n_trajectories=400, horizon=horizon, seed=12)
    attr = SequentialDROPE(n_folds=3, device="cpu", random_state=12).attribute_turns(ds)
    assert attr.turn_effects.shape == (horizon,)
    assert attr.turn_influence_scores.shape == (400, horizon)
    assert attr.additivity_error < 1e-12


def test_simulate_generators_validate_inputs():
    """Synthetic dataset generators reject invalid sample sizes or dimensions."""
    with pytest.raises(ValueError):
        simulate_multiturn_agent(n_trajectories=5)
    with pytest.raises(ValueError):
        simulate_multiturn_agent(horizon=1)
    with pytest.raises(ValueError):
        simulate_multiturn_agent(state_dim=1)
    with pytest.raises(ValueError):
        simulate_guardrail_mediation(n_samples=10)


def test_bundled_openllm_slice_integrity():
    """Bundled Open LLM slice loads cleanly with expected shape and bounded values."""
    sl = load_openllm_slice()
    y = sl["responses"]
    assert y.shape == (60, 1500)
    assert np.all((y >= 0.0) & (y <= 1.0))
    assert len(sl["model_names"]) == 60
    assert len(sl["benchmark_names"]) == 6
    assert sl["benchmark_ids"].shape == (1500,)


@pytest.mark.parametrize("alt", ["greater", "less", "two-sided"])
def test_anytime_auditor_all_alternatives_shape_and_bounds(alt):
    """AnytimeAuditor returns well-formed trajectories for all three alternatives."""
    rng = np.random.default_rng(14)
    stream = rng.uniform(0.2, 0.8, size=250)
    res = AnytimeAuditor(alpha=0.05).audit(stream, null_mean=0.5, alternative=alt)
    assert res.log_e_trajectory.shape == (250,)
    assert res.cs_low.shape == (250,)
    assert res.cs_high.shape == (250,)
    assert np.all(res.cs_low <= res.cs_high)
    assert 0.0 < res.anytime_p_value <= 1.0


def test_guardrail_1d_context_accepted():
    """GuardrailDecomposer accepts 1D context vectors gracefully."""
    rng = np.random.default_rng(31)
    n = 400
    x = rng.normal(0.0, 1.0, size=n)
    a = (rng.random(n) < 1.0 / (1.0 + np.exp(-0.5 * x))).astype(np.int64)
    m = (rng.random(n) < 1.0 / (1.0 + np.exp(-0.5 + 0.8 * a + 0.3 * x))).astype(np.float64)
    y = 0.5 + 0.2 * a - 0.4 * m + 0.2 * x + rng.normal(0.0, 0.2, size=n)
    res = GuardrailDecomposer(n_folds=3, random_state=31).decompose(x, a, m, y)
    assert res.additivity_error < 1e-12


def test_exposure_result_top_flagged_sorting():
    """ExposureAuditResult.top_flagged sorts models in descending log10(e-value)."""
    sl = load_openllm_slice()
    res = ExposureAuditor(device="cpu", random_state=42).audit(
        sl["responses"], sl["model_names"], sl["benchmark_ids"]
    )
    rows = res.top_flagged(10)
    assert len(rows) == 10
    vals = [r["log10_e"] for r in rows]
    assert vals == sorted(vals, reverse=True)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
def test_gpu_end_to_end_ope_and_exposure():
    """CUDA GPU paths match CPU paths end-to-end for SequentialDROPE and ExposureAuditor."""
    ds = simulate_multiturn_agent(n_trajectories=400, horizon=4, seed=99)
    r_cpu = SequentialDROPE(n_folds=3, device="cpu", random_state=99).evaluate(ds)
    r_gpu = SequentialDROPE(n_folds=3, device="cuda:0", random_state=99).evaluate(ds)
    assert r_gpu.device == "cuda:0"
    assert abs(r_gpu.estimate - r_cpu.estimate) < 1e-10
    assert np.max(np.abs(r_gpu.influence_scores - r_cpu.influence_scores)) < 1e-10

    sl = load_openllm_slice()
    e_cpu = ExposureAuditor(device="cpu", random_state=42).audit(
        sl["responses"], sl["model_names"], sl["benchmark_ids"]
    )
    e_gpu = ExposureAuditor(device="cuda:0", random_state=42).audit(
        sl["responses"], sl["model_names"], sl["benchmark_ids"]
    )
    assert e_gpu.device == "cuda:0"
    assert np.max(np.abs(e_gpu.log10_e_values - e_cpu.log10_e_values)) < 1e-10
