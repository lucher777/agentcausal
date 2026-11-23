# agentcausal

**Multi-turn causal attribution, counterfactual guardrail evaluation, and anytime-valid statistical auditing of LLM agents.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](pyproject.toml)

`agentcausal` is a Python library for evaluating and monitoring multi-turn large language model (LLM) agents with established causal and sequential-inference estimators.

## Why `agentcausal`?

Static pass rates and judge scores cannot answer three questions that come up with deployed agents:

1. **Which turn mattered?** Guardrails, retrieval and verifiers are switched on depending on the conversation state, so comparing triggered with untriggered conversations is confounded.
2. **Is a guardrail costing utility?** A guardrail can block unsafe outputs (direct effect) and also cause over-refusal on benign queries (indirect effect).
3. **Can a live dashboard be checked every hour?** Repeated fixed-sample tests inflate false alarms (29.75% after 79 looks in our simulation, at a nominal 5%), and leaderboard gains on hard items may reflect benchmark exposure rather than ability.

## Modules

| Module | Main entry point | What it provides |
|---|---|---|
| `agentcausal.ope` | `SequentialDROPE` | Cross-fitted sequential DR, WDR, DM, IS and WIS off-policy evaluation with standard errors; turn-level effects that sum exactly to the total lift |
| `agentcausal.guardrail` | `GuardrailDecomposer` | Multiply robust natural direct/indirect effect estimation (NDE + NIE = ATE) and a counterfactual safety-utility frontier |
| `agentcausal.anytime` | `AnytimeAuditor` | Betting e-processes and empirical-Bernstein confidence sequences, valid under continuous monitoring and optional stopping |
| `agentcausal.exposure` | `ExposureAuditor` | Leave-one-model-out audit of hard-item excess accuracy in model x item response matrices; flags are a reason for inspection, not proof of contamination |
| `agentcausal.equiv` | `run_equivalence_checks` | Checks of numerical identities (vectorised vs recursive, CPU vs GPU) at tolerance 1e-10 |

## Installation

The package is not yet on PyPI. Install from the repository:

```bash
pip install "git+https://github.com/lucher777/agentcausal.git@v0.1.0"
# optional PyTorch acceleration
pip install "agentcausal[gpu] @ git+https://github.com/lucher777/agentcausal.git@v0.1.0"
```

The core depends only on NumPy, SciPy and scikit-learn. PyTorch is optional; it speeds up large batched e-processes but is slower than NumPy for small off-policy workloads.

## Testing

```bash
pip install -e ".[dev]"
pytest --cov=agentcausal
```

The suite has 63 tests. Line coverage is 100% on a machine with a CUDA GPU and about 92% on CPU-only machines, where the GPU branches are skipped. CI runs on Linux, macOS and Windows with Python 3.9-3.12.

## Quickstart

```python
import numpy as np
from agentcausal import (
    AnytimeAuditor,
    ExposureAuditor,
    GuardrailDecomposer,
    SequentialDROPE,
    load_openllm_slice,
    simulate_guardrail_mediation,
    simulate_multiturn_agent,
)

# 1. Multi-turn doubly-robust OPE and turn-level causal attribution
traj = simulate_multiturn_agent(n_trajectories=2000, horizon=5, seed=42)
ope = SequentialDROPE(n_folds=5, device="auto")
ope_res = ope.evaluate(traj)
turn_attr = ope.attribute_turns(traj)
print(f"DR V(pi_new)={ope_res.estimate:.4f} +/- {ope_res.se:.4f}")
print(f"Turn attributions: {np.round(turn_attr.turn_effects, 4)}")

# 2. Counterfactual guardrail path-specific decomposition (NDE + NIE = ATE)
med = simulate_guardrail_mediation(n_samples=3000, seed=42)
dec = GuardrailDecomposer(n_folds=5)
med_res = dec.decompose(med["contexts"], med["guardrail"], med["mediator"], med["utility"])
print(f"NDE={med_res.nde:+.4f}, NIE={med_res.nie:+.4f}, ATE={med_res.ate:+.4f}")

# 3. Anytime-valid sequential auditing under continuous monitoring
auditor = AnytimeAuditor(alpha=0.05, lower=0.0, upper=1.0)
audit_res = auditor.audit(med["safety"], null_mean=0.50, alternative="greater")
print(f"log10(E)={audit_res.log10_e_value:.2f}, anytime p<={audit_res.anytime_p_value:.4g}")

# 4. Model-free nonparametric hard-tail benchmark exposure audit
sl = load_openllm_slice()
exp_res = ExposureAuditor(device="auto").audit(
    sl["responses"], model_names=sl["model_names"], benchmark_ids=sl["benchmark_ids"]
)
print(exp_res.top_flagged(3))
```

## Command-Line Interface

```bash
agentcausal demo
agentcausal ope --n 2000 --horizon 5
agentcausal audit --n 1000 --true-mean 0.58 --null-mean 0.50
agentcausal exposure --top-k 5
agentcausal equiv --tol 1e-10
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT License. See [LICENSE](LICENSE).
