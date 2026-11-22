"""Tests for AnytimeAuditor betting e-processes and time-uniform confidence sequences."""

from __future__ import annotations

import numpy as np
import pytest

from agentcausal import AnytimeAuditor


def test_anytime_fpr_control_under_continuous_peeking():
    """Ville's inequality keeps empirical anytime FPR <= 0.05 under peeking at every step."""
    rng = np.random.default_rng(1234)
    n_reps = 400
    horizon = 600
    mat = (rng.random((n_reps, horizon)) < 0.50).astype(np.float64)
    aud = AnytimeAuditor(alpha=0.05, lower=0.0, upper=1.0)
    _, max_log_e, _ = aud.batch_eprocess(mat, null_mean=0.50, alternative="greater", device="cpu")
    fpr = float(np.mean(max_log_e >= np.log(1.0 / 0.05)))
    assert fpr <= 0.05


def test_anytime_two_sided_fpr_control_under_peeking():
    """Two-sided mixture betting e-process maintains FPR <= 0.05 under continuous peeking."""
    rng = np.random.default_rng(2345)
    mat = rng.uniform(0.0, 1.0, size=(300, 500))
    aud = AnytimeAuditor(alpha=0.05, lower=0.0, upper=1.0)
    _, max_log_e, _ = aud.batch_eprocess(mat, null_mean=0.50, alternative="two-sided", device="cpu")
    fpr = float(np.mean(max_log_e >= np.log(1.0 / 0.05)))
    assert fpr <= 0.05


def test_anytime_high_power_under_alternative():
    """Betting e-process detects true shift quickly with high e-value and finite stopping time."""
    rng = np.random.default_rng(3456)
    stream = (rng.random(1200) < 0.62).astype(np.float64)
    res = AnytimeAuditor(alpha=0.05).audit(stream, null_mean=0.50, alternative="greater")
    assert res.rejected is True
    assert res.stopping_time is not None and res.stopping_time < 600
    assert res.log10_e_value > 5.0
    assert res.anytime_p_value < 1e-5


def test_anytime_less_alternative():
    """One-sided 'less' alternative detects downward performance regression."""
    rng = np.random.default_rng(4567)
    stream = (rng.random(1000) < 0.38).astype(np.float64)
    res = AnytimeAuditor(alpha=0.05).audit(stream, null_mean=0.50, alternative="less")
    assert res.rejected is True
    assert res.stopping_time is not None


def test_anytime_custom_bounds_rescaling():
    """AnytimeAuditor works natively on bounded influence scores in [-2, 2]."""
    rng = np.random.default_rng(5678)
    scores = np.clip(rng.normal(0.35, 0.5, size=800), -2.0, 2.0)
    aud = AnytimeAuditor(alpha=0.05, lower=-2.0, upper=2.0)
    res = aud.audit(scores, null_mean=0.0, alternative="greater")
    assert res.rejected is True
    assert res.cs_low[-1] > 0.0
    assert res.cs_low[-1] < res.sample_mean < res.cs_high[-1]


def test_time_uniform_confidence_sequence_coverage():
    """Empirical-Bernstein confidence sequence covers true mean uniformly across all t."""
    rng = np.random.default_rng(6789)
    true_mu = 0.65
    aud = AnytimeAuditor(alpha=0.05, lower=0.0, upper=1.0)
    violations = 0
    n_reps = 100
    for _ in range(n_reps):
        stream = (rng.random(500) < true_mu).astype(np.float64)
        lo, hi = aud.confidence_sequence(stream)
        # Monotonic tightening via running intersection
        assert np.all(np.diff(lo) >= -1e-12)
        assert np.all(np.diff(hi) <= 1e-12)
        if np.any(lo > true_mu) or np.any(hi < true_mu):
            violations += 1
    assert violations / n_reps <= 0.06


def test_anytime_vectorized_vs_sequential_parity():
    """Vectorized betting e-process equals explicit loop within 1e-12."""
    rng = np.random.default_rng(88)
    u = rng.uniform(0.0, 1.0, size=350)
    aud = AnytimeAuditor(alpha=0.05)
    for d in ("greater", "less"):
        v = aud._one_sided_log_e_vectorized(u, 0.45, direction=d)
        s = aud._one_sided_log_e_sequential(u, 0.45, direction=d)
        assert np.max(np.abs(v - s)) < 1e-12


def test_anytime_validation_errors():
    """Out-of-bounds observations and invalid parameters raise ValueError."""
    with pytest.raises(ValueError):
        AnytimeAuditor(alpha=0.0)
    with pytest.raises(ValueError):
        AnytimeAuditor(lower=1.0, upper=1.0)
    with pytest.raises(ValueError):
        AnytimeAuditor(max_bet_fraction=1.0)
    aud = AnytimeAuditor(lower=0.0, upper=1.0)
    with pytest.raises(ValueError):
        aud.audit(np.array([0.2, 1.5]), null_mean=0.5)
    with pytest.raises(ValueError):
        aud.audit(np.array([0.2, 0.5]), null_mean=0.5, alternative="bad")
    with pytest.raises(ValueError):
        aud.audit(np.array([]), null_mean=0.5)
