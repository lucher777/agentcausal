# agentcausal Documentation

`agentcausal` is a Python library for **multi-turn causal attribution**, **counterfactual guardrail evaluation**, and **anytime-valid statistical auditing** of large language model (LLM) agents.

## Key Capabilities

- **Sequential Doubly-Robust Off-Policy Evaluation (`SequentialDROPE`)**: Cross-fitted DR, WDR, DM, IS, and WIS estimators for multi-turn conversational trajectories under time-varying confounding, plus exact telescoping turn-level causal attribution.
- **Counterfactual Guardrail Mediation (`GuardrailDecomposer`)**: Multiply-robust efficient influence function (EIF) separation of Natural Direct Effects (NDE) and Natural Indirect Over-Refusal Effects (NIE), plus counterfactual safety-utility Pareto frontier tracing.
- **Anytime-Valid Sequential Auditing (`AnytimeAuditor`)**: Predictable plug-in betting e-processes and empirical-Bernstein confidence sequences with Ville's inequality Type-I error guarantees under arbitrary optional stopping.
- **Nonparametric Exposure Auditing (`ExposureAuditor`)**: Model-free GPU/CPU leave-one-model-out peer-matched hard-tail exposure shift detection with anytime-valid betting e-values.

See the [Tutorial](tutorial.md) for runnable workflows and the [API Reference](api.md) for detailed class signatures.
