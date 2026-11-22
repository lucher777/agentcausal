# Contributing to agentcausal

Thank you for your interest in contributing to `agentcausal`.

## Development Setup

1. Clone the repository and create a Python 3.9+ virtual environment.
2. Install development dependencies (`numpy`, `scipy`, `scikit-learn`, `pytest`, and optionally `torch` for CUDA tests).
3. Run the full test suite and numerical parity checks:

```bash
PYTHONPATH=src pytest tests/ -v
PYTHONPATH=src python -m agentcausal.cli equiv --tol 1e-10
```

## Code Quality Standards

- Every public class, method, and function must have a complete docstring.
- Any new estimator must include unit tests covering theoretical consistency, variance calibration, input validation, and `< 1e-10` numerical parity checks between reference and vectorized/GPU paths.
- Keep CPU execution dependency-light (`numpy`, `scipy`, `scikit-learn`) with `torch` strictly optional.
