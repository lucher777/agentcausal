# agentcausal

**Multi-turn causal attribution, counterfactual guardrail evaluation, and anytime-valid statistical auditing of LLM agents.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](pyproject.toml)

`agentcausal` is a Python library that brings semiparametric causal inference, multiply-robust mediation analysis, and game-theoretic anytime-valid statistical inference to the evaluation and continuous deployment monitoring of multi-turn large language model (LLM) agents.

## Why `agentcausal`?

Standard LLM evaluation harnesses report static pass rates or heuristic LLM-as-a-Judge scores that break down on three statistical fronts critical to deployed multi-turn agents:

1. **Time-varying multi-turn confounding**: Guardrails, safety filters, chain-of-thought scaffolds, and tool-call verifiers are triggered adaptively based on intermediate conversation states $S_t$, confounding naive before/after accuracy comparisons.
2. **Safety-vs-utility path-specific trade-offs**: Guardrails simultaneously block unsafe outputs (direct safety effect) and induce over-refusal or reasoning degradation on benign queries (indirect utility cost).
3. **Optional stopping and benchmark exposure shift**: Continuous deployment monitoring with repeated fixed-horizon $t$-tests inflates false-alarm rates above 30%, while public leaderboards suffer non-exchangeable hard-tail benchmark exposure anomalies.

## Core Modules

| Module | Class | Statistical Guarantee |
|---|---|---|
| `agentcausal.ope` | `SequentialDROPE` | Cross-fitted multi-turn DR / WDR / DM / IS / WIS off-policy evaluation + exact telescoping turn-level attribution ($\sum_{t=1}^H \phi_t = V(\pi_{\text{new}}) - V(\pi_{\text{base}})$) |
| `agentcausal.guardrail` | `GuardrailDecomposer` | Multiply-robust EIF decomposition ($\mathrm{NDE} + \mathrm{NIE} = \mathrm{ATE}$) + counterfactual safety-utility Pareto frontier |
| `agentcausal.anytime` | `AnytimeAuditor` | Predictable plug-in betting e-processes & empirical-Bernstein confidence sequences with Ville's inequality $\alpha$-level anytime-valid FPR control |
| `agentcausal.exposure` | `ExposureAuditor` | Model-free GPU/CPU leave-one-model-out peer-matched exposure shift & anytime-valid e-value auditing (zero parametric IRT assumptions) |
| `agentcausal.equiv` | `run_equivalence_checks` | Strict $<10^{-10}$ numerical parity across recursive, vectorized, CPU, and CUDA GPU float64 paths |

## Installation

```bash
pip install agentcausal
# Optional PyTorch CUDA GPU acceleration:
pip install "agentcausal[gpu]"
```

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
