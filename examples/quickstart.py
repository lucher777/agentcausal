"""End-to-end quickstart demonstrating all four modules of ``agentcausal``."""

from __future__ import annotations

import numpy as np

from agentcausal import (
    AnytimeAuditor,
    ExposureAuditor,
    GuardrailDecomposer,
    SequentialDROPE,
    load_openllm_slice,
    run_equivalence_checks,
    simulate_guardrail_mediation,
    simulate_multiturn_agent,
)


def main() -> None:
    """Run all four statistical auditing modules and print concise summaries."""
    print("1. Sequential Doubly-Robust Off-Policy Evaluation & Turn Attribution")
    traj = simulate_multiturn_agent(n_trajectories=1500, horizon=5, seed=42)
    ope = SequentialDROPE(n_folds=5, device="auto", random_state=42)
    res_ope = ope.evaluate(traj)
    attr = ope.attribute_turns(traj)
    print(
        f"   DR={res_ope.estimate:.4f} (SE={res_ope.se:.4f}, true={traj.metadata['true_v_new']:.4f}) "
        f"on {res_ope.device}"
    )
    print(f"   Turn attributions: {np.round(attr.turn_effects, 4)} (err={attr.additivity_error:.2e})")

    print("2. Counterfactual Guardrail Mediation & Safety-Utility Frontier")
    med = simulate_guardrail_mediation(n_samples=2000, seed=42)
    dec = GuardrailDecomposer(n_folds=5, random_state=42)
    res_med = dec.decompose(med["contexts"], med["guardrail"], med["mediator"], med["utility"])
    print(
        f"   NDE={res_med.nde:+.4f} (true={med['true_nde']:+.4f}), "
        f"NIE={res_med.nie:+.4f} (true={med['true_nie']:+.4f}), "
        f"ATE={res_med.ate:+.4f}"
    )
    frontier = dec.pareto_frontier(
        med["contexts"],
        med["guardrail"],
        med["safety"],
        med["utility"],
        med["risk_scores"],
        thresholds=np.linspace(0.15, 0.85, 8),
    )
    n_opt = sum(1 for pt in frontier if pt.is_pareto_optimal)
    print(f"   Pareto-optimal threshold policies: {n_opt}/{len(frontier)}")

    print("3. Anytime-Valid Sequential Deployment Auditing")
    aud = AnytimeAuditor(alpha=0.05, lower=0.0, upper=1.0)
    res_aud = aud.audit(med["safety"], null_mean=0.50, alternative="greater")
    print(
        f"   log10(E)={res_aud.log10_e_value:.2f}, stopping_time={res_aud.stopping_time}, "
        f"CS=[{res_aud.cs_low[-1]:.3f}, {res_aud.cs_high[-1]:.3f}]"
    )

    print("4. Nonparametric Open LLM Hard-Tail Exposure Audit")
    sl = load_openllm_slice()
    res_exp = ExposureAuditor(device="auto", random_state=42).audit(
        sl["responses"], model_names=sl["model_names"], benchmark_ids=sl["benchmark_ids"]
    )
    top3 = res_exp.top_flagged(3)
    for row in top3:
        print(
            f"   #{row['rank']} {row['model']} delta={row['delta']:+.3f} "
            f"log10(E)={row['log10_e']:.2f} flagged={row['flagged']}"
        )

    print("5. Strict Numerical Parity Verification")
    eq = run_equivalence_checks(tolerance=1e-10)
    print(f"   Passed={eq.passed} (device={eq.device_name}, max_err={max(eq.checks.values()):.2e})")


if __name__ == "__main__":
    main()
