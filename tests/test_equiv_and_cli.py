"""Tests for numerical parity (equiv.py), GPU parity, and CLI subcommands."""

from __future__ import annotations

import json

import pytest

from agentcausal import run_equivalence_checks
from agentcausal.cli import main as cli_main

try:
    import torch

    HAS_CUDA = bool(torch.cuda.is_available())
except ImportError:
    HAS_CUDA = False


def test_equivalence_checks_pass_under_1e_10():
    """All equivalence checks pass strictly below 1e-10."""
    rep = run_equivalence_checks(tolerance=1e-10, seed=2026)
    assert rep.passed is True
    assert rep.checks["dr_vectorized_vs_recursive"] < 1e-12
    assert rep.checks["turn_attribution_additivity"] < 1e-12
    assert rep.checks["mediation_additivity"] < 1e-12
    assert rep.checks["eprocess_vectorized_vs_sequential"] < 1e-12
    if HAS_CUDA:
        assert rep.device_checked is True
        assert rep.checks["dr_gpu_vs_cpu"] < 1e-10
        assert rep.checks["eprocess_gpu_vs_cpu"] < 1e-10
        assert rep.checks["exposure_gpu_vs_cpu"] < 1e-10


@pytest.mark.parametrize(
    "argv",
    [
        ["demo", "--device", "cpu", "--seed", "7"],
        ["ope", "--n", "200", "--horizon", "3", "--device", "cpu"],
        ["audit", "--n", "300", "--true-mean", "0.60", "--null-mean", "0.50"],
        ["exposure", "--top-k", "3", "--device", "cpu"],
        ["equiv", "--tol", "1e-10"],
    ],
)
def test_cli_subcommands_return_valid_json(argv, capsys):
    """Every CLI subcommand executes cleanly and emits valid JSON."""
    rc = cli_main(argv)
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
