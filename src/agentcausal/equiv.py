"""Numerical parity and algebraic identity verification suite for ``agentcausal``."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .anytime import AnytimeAuditor
from .datasets import (
    load_openllm_slice,
    simulate_guardrail_mediation,
    simulate_multiturn_agent,
)
from .exposure import ExposureAuditor
from .guardrail import GuardrailDecomposer
from .ope import SequentialDROPE
from .types import EquivalenceReport

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def run_equivalence_checks(
    tolerance: float = 1e-10,
    seed: int = 2026,
) -> EquivalenceReport:
    """Verify exact numerical parity across recursive, vectorized, CPU, and GPU paths.

    Executes seven deterministic checks:

    1. ``dr_vectorized_vs_recursive``: Closed-form vectorized DR scores vs. backward
       recursive Bellman DR formula.
    2. ``turn_attribution_additivity``: Exact telescoping sum identity
       ``|sum_t phi_t - (V_new - V_base)|``.
    3. ``mediation_additivity``: Exact path-specific identity ``|(NDE + NIE) - ATE|``.
    4. ``eprocess_vectorized_vs_sequential``: Vectorized predictable betting log-wealth
       vs. explicit sequential loop.
    5. ``dr_gpu_vs_cpu``: PyTorch CUDA float64 vs. NumPy CPU DR-OPE scores (when GPU available).
    6. ``eprocess_gpu_vs_cpu``: PyTorch CUDA float64 vs. NumPy CPU batched e-process.
    7. ``exposure_gpu_vs_cpu``: PyTorch CUDA float64 vs. NumPy CPU ``ExposureAuditor``
       on the bundled Open LLM exposure slice.

    Args:
        tolerance: Maximum acceptable absolute discrepancy (default ``1e-10``).
        seed: Random seed for synthetic parity inputs.

    Returns:
        ``EquivalenceReport`` detailing every discrepancy and overall pass status.
    """
    rng = np.random.default_rng(seed)
    checks: Dict[str, float] = {}

    # 1. Vectorized vs recursive DR scores
    n, h = 256, 5
    rewards = rng.normal(1.0, 0.5, size=(n, h))
    rho = rng.uniform(0.3, 2.2, size=(n, h))
    q_obs = rng.normal(2.0, 0.7, size=(n, h))
    v_hat = rng.normal(2.1, 0.7, size=(n, h))
    dr_vec, wdr_cpu, dm_cpu, is_cpu, wis_cpu, _ = SequentialDROPE.compute_dr_scores_vectorized(
        rewards=rewards,
        rho_step=rho,
        q_obs=q_obs,
        v_hat=v_hat,
        discount=0.95,
        device="cpu",
    )
    dr_rec = SequentialDROPE.compute_dr_scores_recursive(
        rewards=rewards,
        rho_step=rho,
        q_obs=q_obs,
        v_hat=v_hat,
        discount=0.95,
    )
    checks["dr_vectorized_vs_recursive"] = float(np.max(np.abs(dr_vec - dr_rec)))

    # 2. Turn-level attribution telescoping additivity
    traj = simulate_multiturn_agent(n_trajectories=240, horizon=4, seed=seed)
    ope = SequentialDROPE(n_folds=3, device="cpu", random_state=seed)
    attr = ope.attribute_turns(traj)
    checks["turn_attribution_additivity"] = float(attr.additivity_error)

    # 3. Guardrail mediation NDE + NIE == ATE additivity
    med = simulate_guardrail_mediation(n_samples=300, seed=seed)
    decomp = GuardrailDecomposer(n_folds=3, random_state=seed).decompose(
        contexts=med["contexts"],
        guardrail=med["guardrail"],
        mediator=med["mediator"],
        outcome=med["utility"],
    )
    checks["mediation_additivity"] = float(decomp.additivity_error)

    # 4. Predictable betting e-process vectorized vs sequential loop
    auditor = AnytimeAuditor(alpha=0.05, lower=0.0, upper=1.0)
    stream = rng.uniform(0.0, 1.0, size=400)
    vec_log_e = auditor._one_sided_log_e_vectorized(stream, 0.45, direction="greater")
    seq_log_e = auditor._one_sided_log_e_sequential(stream, 0.45, direction="greater")
    checks["eprocess_vectorized_vs_sequential"] = float(np.max(np.abs(vec_log_e - seq_log_e)))

    # 5-7. GPU vs CPU float64 parity checks when CUDA is available
    has_cuda = bool(torch is not None and torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(0) if has_cuda else "cpu"

    if has_cuda:
        dr_gpu, wdr_gpu, dm_gpu, is_gpu, wis_gpu, _ = (
            SequentialDROPE.compute_dr_scores_vectorized(
                rewards=rewards,
                rho_step=rho,
                q_obs=q_obs,
                v_hat=v_hat,
                discount=0.95,
                device="cuda:0",
            )
        )
        checks["dr_gpu_vs_cpu"] = float(
            max(
                np.max(np.abs(dr_gpu - dr_vec)),
                np.max(np.abs(wdr_gpu - wdr_cpu)),
                np.max(np.abs(dm_gpu - dm_cpu)),
                np.max(np.abs(is_gpu - is_cpu)),
                np.max(np.abs(wis_gpu - wis_cpu)),
            )
        )

        mat = rng.uniform(0.0, 1.0, size=(32, 300))
        f_cpu, m_cpu, _ = auditor.batch_eprocess(
            mat, null_mean=0.5, alternative="two-sided", device="cpu"
        )
        f_gpu, m_gpu, _ = auditor.batch_eprocess(
            mat, null_mean=0.5, alternative="two-sided", device="cuda:0"
        )
        checks["eprocess_gpu_vs_cpu"] = float(
            max(np.max(np.abs(f_gpu - f_cpu)), np.max(np.abs(m_gpu - m_cpu)))
        )

        sl = load_openllm_slice()
        exp_cpu = ExposureAuditor(device="cpu", random_state=seed).audit(
            responses=sl["responses"],
            model_names=sl["model_names"],
            benchmark_ids=sl["benchmark_ids"],
        )
        exp_gpu = ExposureAuditor(device="cuda:0", random_state=seed).audit(
            responses=sl["responses"],
            model_names=sl["model_names"],
            benchmark_ids=sl["benchmark_ids"],
        )
        checks["exposure_gpu_vs_cpu"] = float(
            max(
                np.max(np.abs(exp_gpu.delta - exp_cpu.delta)),
                np.max(np.abs(exp_gpu.log10_e_values - exp_cpu.log10_e_values)),
            )
        )

    passed = all(val <= tolerance for val in checks.values())
    return EquivalenceReport(
        passed=passed,
        tolerance=float(tolerance),
        checks=checks,
        device_checked=has_cuda,
        device_name=device_name,
    )
