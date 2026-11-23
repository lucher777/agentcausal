# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kang Liu
"""Tests for ExposureAuditor nonparametric hard-tail benchmark exposure auditing."""

from __future__ import annotations

import numpy as np
import pytest

from agentcausal import ExposureAuditor, load_openllm_slice


def test_exposure_auditor_bundled_real_openllm_slice():
    """Bundled real Open LLM slice flags Aquila2-34B and yayi2-30b-llama as top anomalies."""
    sl = load_openllm_slice()
    res = ExposureAuditor(hard_threshold=0.20, device="cpu", random_state=42).audit(
        responses=sl["responses"],
        model_names=sl["model_names"],
        benchmark_ids=sl["benchmark_ids"],
    )
    assert res.n_total_items == 1500
    assert res.n_hard_items > 100
    top3 = [row["model"] for row in res.top_flagged(3)]
    assert "BAAI/Aquila2-34B" in top3
    assert "cognitivecomputations/yayi2-30b-llama" in top3
    top_row = res.top_flagged(1)[0]
    assert top_row["flagged"] is True
    assert top_row["log10_e"] > 2.0


def test_exposure_auditor_detects_synthetic_contaminated_model():
    """ExposureAuditor flags a synthetically exposed model while keeping clean models null."""
    rng = np.random.default_rng(707)
    m, j = 40, 800
    ability = rng.normal(0.0, 1.0, size=(m, 1))
    diff = rng.normal(0.0, 1.5, size=(1, j))
    probs = 1.0 / (1.0 + np.exp(-(ability - diff)))
    y = (rng.random((m, j)) < probs).astype(np.float64)

    # Inject hard-tail contamination into model 0 only
    hard = y.mean(axis=0) <= 0.20
    y[0, hard] = np.where(rng.random(int(hard.sum())) < 0.35, 1.0, y[0, hard])

    res = ExposureAuditor(hard_threshold=0.20, device="cpu", random_state=707).audit(y)
    assert res.flagged[0]
    assert res.log10_e_values[0] > 4.0
    assert res.delta[0] > 0.15
    # Clean models maintain low false-positive rate
    clean_fpr = float(res.flagged[1:].mean())
    assert clean_fpr <= 0.10


def test_exposure_auditor_null_calibration_without_contamination():
    """Under a clean logistic capability matrix without exposure, FPR stays small."""
    rng = np.random.default_rng(808)
    m, j = 50, 900
    ability = rng.normal(0.0, 1.0, size=(m, 1))
    diff = rng.normal(0.0, 1.4, size=(1, j))
    probs = 1.0 / (1.0 + np.exp(-(ability - diff)))
    y = (rng.random((m, j)) < probs).astype(np.float64)

    res = ExposureAuditor(hard_threshold=0.20, device="cpu", random_state=808).audit(y)
    assert float(res.flagged.mean()) <= 0.10


def test_exposure_auditor_without_benchmark_ids():
    """ExposureAuditor works cleanly when benchmark_ids is omitted."""
    sl = load_openllm_slice()
    res = ExposureAuditor(device="cpu", random_state=42).audit(
        responses=sl["responses"], model_names=sl["model_names"]
    )
    assert len(res.model_names) == 60
    assert np.all(np.isfinite(res.log10_e_values))
    assert np.all((res.anytime_p_values >= 0.0) & (res.anytime_p_values <= 1.0))


def test_exposure_auditor_validation_errors():
    """Invalid thresholds, NaNs, or malformed shapes raise ValueError."""
    with pytest.raises(ValueError):
        ExposureAuditor(hard_threshold=0.95)
    with pytest.raises(ValueError):
        ExposureAuditor(n_difficulty_bins=1)
    with pytest.raises(ValueError):
        ExposureAuditor(ridge_alpha=0.0)
    aud = ExposureAuditor(device="cpu")
    with pytest.raises(ValueError):
        aud.audit(np.zeros((3, 50)))
    bad = np.full((10, 40), 0.5)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        aud.audit(bad)
    with pytest.raises(ValueError):
        aud.audit(np.full((10, 40), 1.5))
