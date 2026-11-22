# API Reference

## Core Estimators

### `agentcausal.SequentialDROPE`
- `SequentialDROPE(n_folds=5, propensity_clip=0.02, ridge_alpha=1.0, alpha=0.05, device="auto", random_state=42)`
- `.evaluate(dataset, target_action_probs=None, method="dr") -> DROPEResult`
- `.attribute_turns(dataset, new_action_probs=None, base_action_probs=None) -> TurnAttributionResult`
- `.compute_dr_scores_vectorized(rewards, rho_step, q_obs, v_hat, discount=1.0, device="cpu")`
- `.compute_dr_scores_recursive(rewards, rho_step, q_obs, v_hat, discount=1.0)`

### `agentcausal.GuardrailDecomposer`
- `GuardrailDecomposer(n_folds=5, propensity_clip=0.02, ridge_alpha=1.0, alpha=0.05, random_state=42)`
- `.decompose(contexts, guardrail, mediator, outcome) -> GuardrailMediationResult`
- `.pareto_frontier(contexts, guardrail, safety_outcome, utility_outcome, risk_scores, thresholds) -> List[ParetoPoint]`

### `agentcausal.AnytimeAuditor`
- `AnytimeAuditor(alpha=0.05, lower=0.0, upper=1.0, prior_strength=5.0, max_bet_fraction=0.75)`
- `.audit(values, null_mean, alternative="greater") -> AnytimeAuditResult`
- `.confidence_sequence(values, alpha=None) -> Tuple[np.ndarray, np.ndarray]`
- `.batch_eprocess(matrix, null_mean, alternative="greater", device="auto")`

### `agentcausal.ExposureAuditor`
- `ExposureAuditor(hard_threshold=0.20, n_difficulty_bins=8, ridge_alpha=10.0, alpha=0.05, device="auto", random_state=42)`
- `.audit(responses, model_names=None, benchmark_ids=None) -> ExposureAuditResult`

## Verification & Datasets

- `agentcausal.run_equivalence_checks(tolerance=1e-10, seed=2026) -> EquivalenceReport`
- `agentcausal.simulate_multiturn_agent(...) -> TrajectoryDataset`
- `agentcausal.simulate_guardrail_mediation(...) -> Dict[str, object]`
- `agentcausal.load_openllm_slice() -> Dict[str, np.ndarray]`
