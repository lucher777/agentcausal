"""Command-line interface for ``agentcausal``."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

import numpy as np

from .anytime import AnytimeAuditor
from .datasets import (
    load_openllm_slice,
    simulate_guardrail_mediation,
    simulate_multiturn_agent,
)
from .equiv import run_equivalence_checks
from .exposure import ExposureAuditor
from .guardrail import GuardrailDecomposer
from .ope import SequentialDROPE


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``agentcausal`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="agentcausal",
        description=(
            "Multi-turn causal attribution, counterfactual guardrail evaluation, "
            "and anytime-valid statistical auditing for LLM agents."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run an end-to-end demonstration of all four modules.")
    p_demo.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_demo.add_argument("--device", type=str, default="auto", help="Compute device (auto/cpu/cuda).")

    p_ope = sub.add_parser("ope", help="Run sequential DR-OPE and turn-level attribution.")
    p_ope.add_argument("--n", type=int, default=1500, help="Number of trajectories.")
    p_ope.add_argument("--horizon", type=int, default=5, help="Conversation horizon H.")
    p_ope.add_argument("--device", type=str, default="auto", help="Compute device.")
    p_ope.add_argument("--seed", type=int, default=42, help="Random seed.")

    p_audit = sub.add_parser("audit", help="Run an anytime-valid sequential e-value audit.")
    p_audit.add_argument("--n", type=int, default=1000, help="Stream length.")
    p_audit.add_argument("--true-mean", type=float, default=0.58, help="True stream mean.")
    p_audit.add_argument("--null-mean", type=float, default=0.50, help="Null boundary mu_0.")
    p_audit.add_argument("--alpha", type=float, default=0.05, help="Anytime Type-I error bound.")
    p_audit.add_argument("--seed", type=int, default=42, help="Random seed.")

    p_exp = sub.add_parser("exposure", help="Run nonparametric exposure audit on bundled Open LLM slice.")
    p_exp.add_argument("--top-k", type=int, default=5, help="Number of top flagged models to print.")
    p_exp.add_argument("--device", type=str, default="auto", help="Compute device.")
    p_exp.add_argument("--seed", type=int, default=42, help="Random seed.")

    p_eq = sub.add_parser("equiv", help="Verify strict <1e-10 numerical parity across routines.")
    p_eq.add_argument("--tol", type=float, default=1e-10, help="Parity tolerance.")
    p_eq.add_argument("--seed", type=int, default=2026, help="Random seed.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute the ``agentcausal`` CLI and print JSON-formatted output.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Zero exit status on success, non-zero if equivalence verification fails.
    """
    args = build_parser().parse_args(argv)

    if args.command == "demo":
        traj = simulate_multiturn_agent(n_trajectories=800, horizon=4, seed=args.seed)
        ope = SequentialDROPE(n_folds=4, device=args.device, random_state=args.seed)
        res_ope = ope.evaluate(traj)
        attr = ope.attribute_turns(traj)

        med = simulate_guardrail_mediation(n_samples=1000, seed=args.seed)
        res_med = GuardrailDecomposer(n_folds=4, random_state=args.seed).decompose(
            med["contexts"], med["guardrail"], med["mediator"], med["utility"]
        )

        sl = load_openllm_slice()
        res_exp = ExposureAuditor(device=args.device, random_state=args.seed).audit(
            sl["responses"], sl["model_names"], sl["benchmark_ids"]
        )
        payload = {
            "ope_dr": round(res_ope.estimate, 4),
            "ope_true": round(float(traj.metadata["true_v_new"]), 4),
            "turn_additivity_error": attr.additivity_error,
            "guardrail_nde": round(res_med.nde, 4),
            "guardrail_nie": round(res_med.nie, 4),
            "guardrail_ate": round(res_med.ate, 4),
            "exposure_top3": res_exp.top_flagged(3),
            "device": res_exp.device,
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "ope":
        traj = simulate_multiturn_agent(
            n_trajectories=args.n, horizon=args.horizon, seed=args.seed
        )
        ope = SequentialDROPE(device=args.device, random_state=args.seed)
        res = ope.evaluate(traj)
        attr = ope.attribute_turns(traj)
        payload = {
            "estimates": {k: round(v, 5) for k, v in res.estimates.items()},
            "true_v_new": round(float(traj.metadata["true_v_new"]), 5),
            "se_dr": round(res.se, 5),
            "turn_effects": [round(float(x), 5) for x in attr.turn_effects],
            "additivity_error": attr.additivity_error,
            "device": res.device,
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "audit":
        rng = np.random.default_rng(args.seed)
        stream = (rng.random(args.n) < args.true_mean).astype(np.float64)
        res = AnytimeAuditor(alpha=args.alpha).audit(stream, null_mean=args.null_mean)
        payload = {
            "sample_mean": round(res.sample_mean, 5),
            "log10_e_value": round(res.log10_e_value, 4),
            "anytime_p_value": res.anytime_p_value,
            "rejected": res.rejected,
            "stopping_time": res.stopping_time,
            "final_cs": [round(float(res.cs_low[-1]), 5), round(float(res.cs_high[-1]), 5)],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "exposure":
        sl = load_openllm_slice()
        res = ExposureAuditor(device=args.device, random_state=args.seed).audit(
            sl["responses"], sl["model_names"], sl["benchmark_ids"]
        )
        payload = {
            "device": res.device,
            "elapsed_seconds": round(res.elapsed_seconds, 4),
            "n_models": len(res.model_names),
            "n_hard_items": res.n_hard_items,
            "top_flagged": res.top_flagged(args.top_k),
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "equiv":
        rep = run_equivalence_checks(tolerance=args.tol, seed=args.seed)
        payload = {
            "passed": rep.passed,
            "tolerance": rep.tolerance,
            "device_checked": rep.device_checked,
            "device_name": rep.device_name,
            "checks": rep.checks,
        }
        print(json.dumps(payload, indent=2))
        return 0 if rep.passed else 1

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
