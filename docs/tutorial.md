# Tutorial

## 1. Multi-Turn Doubly-Robust Off-Policy Evaluation & Turn Attribution

```python
from agentcausal import SequentialDROPE, simulate_multiturn_agent

dataset = simulate_multiturn_agent(n_trajectories=2000, horizon=5, seed=42)
evaluator = SequentialDROPE(n_folds=5, device="auto", random_state=42)
res = evaluator.evaluate(dataset)
print("DR estimate:", res.estimate, "SE:", res.se, "95% CI:", (res.ci_low, res.ci_high))

attr = evaluator.attribute_turns(dataset)
print("Per-turn causal lift:", attr.turn_effects)
print("Exact additivity error:", attr.additivity_error)
```

## 2. Guardrail Path-Specific Decomposition & Safety-Utility Frontier

```python
import numpy as np
from agentcausal import GuardrailDecomposer, simulate_guardrail_mediation

med = simulate_guardrail_mediation(n_samples=3000, seed=42)
dec = GuardrailDecomposer(n_folds=5, random_state=42)
res = dec.decompose(med["contexts"], med["guardrail"], med["mediator"], med["utility"])
print(f"ATE={res.ate:+.4f}, NDE={res.nde:+.4f}, NIE={res.nie:+.4f}")

frontier = dec.pareto_frontier(
    contexts=med["contexts"],
    guardrail=med["guardrail"],
    safety_outcome=med["safety"],
    utility_outcome=med["utility"],
    risk_scores=med["risk_scores"],
    thresholds=np.linspace(0.1, 0.9, 9),
)
for pt in frontier:
    print(f"tau={pt.threshold:.2f} safety={pt.safety_rate:.3f} utility={pt.utility_rate:.3f}")
```

## 3. Continuous Deployment Monitoring with Anytime-Valid e-Values

```python
from agentcausal import AnytimeAuditor

auditor = AnytimeAuditor(alpha=0.05, lower=0.0, upper=1.0)
res = auditor.audit(med["safety"], null_mean=0.50, alternative="greater")
print("Rejected:", res.rejected, "Stopping time:", res.stopping_time)
print("Final 95% time-uniform CS:", (res.cs_low[-1], res.cs_high[-1]))
```

## 4. Nonparametric Benchmark Exposure Auditing

```python
from agentcausal import ExposureAuditor, load_openllm_slice

sl = load_openllm_slice()
exp = ExposureAuditor(hard_threshold=0.20, device="auto").audit(
    responses=sl["responses"],
    model_names=sl["model_names"],
    benchmark_ids=sl["benchmark_ids"],
)
for row in exp.top_flagged(5):
    print(row["rank"], row["model"], f"delta={row['delta']:+.3f}", f"log10E={row['log10_e']:.2f}")
```
