"""agentcausal: Multi-turn causal attribution, counterfactual guardrail evaluation,
and anytime-valid statistical auditing of LLM agents.
"""

from __future__ import annotations

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
from .types import (
    AnytimeAuditResult,
    DROPEResult,
    EquivalenceReport,
    ExposureAuditResult,
    GuardrailMediationResult,
    ParetoPoint,
    TrajectoryDataset,
    TurnAttributionResult,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AnytimeAuditResult",
    "AnytimeAuditor",
    "DROPEResult",
    "EquivalenceReport",
    "ExposureAuditResult",
    "ExposureAuditor",
    "GuardrailDecomposer",
    "GuardrailMediationResult",
    "ParetoPoint",
    "SequentialDROPE",
    "TrajectoryDataset",
    "TurnAttributionResult",
    "load_openllm_slice",
    "run_equivalence_checks",
    "simulate_guardrail_mediation",
    "simulate_multiturn_agent",
]
