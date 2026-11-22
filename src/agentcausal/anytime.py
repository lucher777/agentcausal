"""Anytime-valid sequential betting e-processes and time-uniform confidence sequences.

Implements predictable plug-in betting e-processes and empirical-Bernstein time-uniform
confidence sequences following Waudby-Smith & Ramdas (2024, JRSS-B) and Ville's
martingale inequality, supporting arbitrary optional stopping and continuous deployment
monitoring of LLM agents.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .types import AnytimeAuditResult

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _resolve_device(device: str) -> str:
    """Resolve compute device string to ``'cuda:0'`` or ``'cpu'``."""
    if device == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    if device.startswith("cuda"):
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but PyTorch CUDA is unavailable.")
        return "cuda:0" if device == "cuda" else device
    if device == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device specification: {device!r}.")


class AnytimeAuditor:
    """Anytime-valid sequential auditor using predictable betting e-processes.

    Monitors a stream of bounded observations or bounded causal influence scores
    ``X_1, X_2, ...`` lying in ``[lower, upper]`` and tests ``H_0: E[X_t | F_{t-1}] <= mu_0``
    (or ``>= mu_0`` / ``== mu_0``) with exact Type-I error control at level ``alpha``
    simultaneously across all sample sizes ``n >= 1`` under arbitrary optional stopping.

    Args:
        alpha: Target anytime-valid Type-I error probability in ``(0, 1)``.
        lower: Known lower bound of observations (default ``0.0``).
        upper: Known upper bound of observations (default ``1.0``).
        prior_strength: Pseudo-count stabilizing early predictable bet sizing.
        max_bet_fraction: Upper cap in ``(0, 1)`` on the safe betting fraction so that
            every wealth factor ``1 + lambda_t (U_t - u_0)`` remains strictly positive.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        lower: float = 0.0,
        upper: float = 1.0,
        prior_strength: float = 5.0,
        max_bet_fraction: float = 0.75,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
        if upper <= lower:
            raise ValueError(f"upper ({upper}) must exceed lower ({lower}).")
        if prior_strength <= 0.0:
            raise ValueError(f"prior_strength must be positive, got {prior_strength}.")
        if not (0.0 < max_bet_fraction < 1.0):
            raise ValueError(
                f"max_bet_fraction must lie in (0, 1), got {max_bet_fraction}."
            )
        self.alpha = float(alpha)
        self.lower = float(lower)
        self.upper = float(upper)
        self.prior_strength = float(prior_strength)
        self.max_bet_fraction = float(max_bet_fraction)

    def _rescale(self, values: np.ndarray) -> np.ndarray:
        """Linearly map observations from ``[lower, upper]`` into ``[0, 1]``."""
        arr = np.asarray(values, dtype=np.float64)
        scaled = (arr - self.lower) / (self.upper - self.lower)
        if np.any(scaled < -1e-9) or np.any(scaled > 1.0 + 1e-9):
            raise ValueError(
                f"Observations lie outside [{self.lower}, {self.upper}]: "
                f"min={arr.min():.4f}, max={arr.max():.4f}."
            )
        return np.clip(scaled, 0.0, 1.0)

    def _one_sided_log_e_vectorized(
        self,
        u: np.ndarray,
        u0: float,
        direction: str,
    ) -> np.ndarray:
        """Compute exact predictable GRAPA-style betting log-wealth trajectory.

        For ``direction == 'greater'`` (testing ``H_0: E[U] <= u0``), we use the
        predictable Kelly-truncated betting fraction:
        ``lambda_t = clip(mu_prev / (s2_prev + 1e-8), 0, c / u0)``
        where ``mu_prev`` and ``s2_prev`` are computed strictly from ``U_1, ..., U_{t-1}``.
        Because ``U_t - u0 >= -u0``, ``1 + lambda_t (U_t - u0) >= 1 - c > 0`` always holds,
        and under ``H_0: E[U_t - u0 | F_{t-1}] <= 0``, the product ``E_n`` is a nonnegative
        supermartingale with ``E_0 = 1``.

        Args:
            u: Rescaled observations in ``[0, 1]`` of shape ``(..., n)``.
            u0: Null boundary in ``(0, 1)``.
            direction: Either ``'greater'`` or ``'less'``.

        Returns:
            Cumulative natural-log wealth trajectory of the same shape as ``u``.
        """
        if not (0.0 < u0 < 1.0):
            raise ValueError(f"Rescaled null_mean must lie strictly in (0, 1), got {u0}.")
        if direction == "greater":
            x = u - u0
            lam_cap = self.max_bet_fraction / u0
        elif direction == "less":
            x = u0 - u
            lam_cap = self.max_bet_fraction / (1.0 - u0)
        else:
            raise ValueError(f"Invalid direction {direction!r}; use 'greater' or 'less'.")

        n = x.shape[-1]
        if n == 0:
            return np.zeros_like(x, dtype=np.float64)

        csum = np.cumsum(x, axis=-1)
        csq = np.cumsum(x * x, axis=-1)
        zeros = np.zeros(x.shape[:-1] + (1,), dtype=np.float64)
        prev_sum = np.concatenate([zeros, csum[..., :-1]], axis=-1)
        prev_sq = np.concatenate([zeros, csq[..., :-1]], axis=-1)
        prev_n = np.arange(n, dtype=np.float64)

        p0 = self.prior_strength
        mu_prev = prev_sum / (prev_n + p0)
        s2_prev = (prev_sq + 0.25 * p0) / (prev_n + p0)
        lam = np.clip(mu_prev / (s2_prev + 1e-8), 0.0, lam_cap)
        return np.cumsum(np.log1p(lam * x), axis=-1)

    def _one_sided_log_e_sequential(
        self,
        u: np.ndarray,
        u0: float,
        direction: str,
    ) -> np.ndarray:
        """Exact step-by-step reference loop for predictable betting log-wealth.

        Used to verify numerical parity against vectorized and GPU implementations.
        """
        if u.ndim != 1:
            raise ValueError("Sequential reference expects a 1D array.")
        x = (u - u0) if direction == "greater" else (u0 - u)
        lam_cap = (
            self.max_bet_fraction / u0
            if direction == "greater"
            else self.max_bet_fraction / (1.0 - u0)
        )
        p0 = self.prior_strength
        n = x.shape[0]
        log_e = np.empty(n, dtype=np.float64)
        running_sum = 0.0
        running_sq = 0.0
        running_log_e = 0.0
        for t in range(n):
            mu_prev = running_sum / (t + p0)
            s2_prev = (running_sq + 0.25 * p0) / (t + p0)
            lam_t = float(np.clip(mu_prev / (s2_prev + 1e-8), 0.0, lam_cap))
            running_log_e += float(np.log1p(lam_t * x[t]))
            log_e[t] = running_log_e
            running_sum += float(x[t])
            running_sq += float(x[t] * x[t])
        return log_e

    def confidence_sequence(
        self,
        values: np.ndarray,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a time-uniform empirical-Bernstein confidence sequence.

        Implements the predictable plug-in empirical-Bernstein confidence sequence
        (Waudby-Smith & Ramdas, 2024, Theorem 2) guaranteeing
        ``P(forall n >= 1: mu* in [L_n, U_n]) >= 1 - alpha`` for bounded observations
        in ``[lower, upper]``.

        Args:
            values: 1D array of sequential observations in ``[lower, upper]``.
            alpha: Optional override for the miscoverage level.

        Returns:
            Tuple ``(cs_low, cs_high)`` of shape ``(n,)`` mapped back to ``[lower, upper]``.
        """
        a = self.alpha if alpha is None else float(alpha)
        u = self._rescale(values)
        n = u.shape[0]
        if n == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        t = np.arange(1, n + 1, dtype=np.float64)
        csum = np.cumsum(u)
        mu_hat = (0.5 + csum) / (t + 1.0)
        mu_prev = np.concatenate([[0.5], mu_hat[:-1]])

        # Predictable empirical variance proxy v_t = (U_t - mu_{t-1})^2
        v_t = (u - mu_prev) ** 2
        sigma2_prev = np.concatenate(
            [[0.25], (0.25 + np.cumsum(v_t)[:-1]) / (t[:-1] + 1.0)]
        )
        # Predictable empirical-Bernstein weights lambda_t (bounded in (0, 0.5])
        log_term = np.log(2.0 / a)
        lam = np.minimum(
            0.5,
            np.sqrt(2.0 * log_term / (sigma2_prev * t * np.log(t + 1.0) + 1e-8)),
        )
        # Exact psi_E(lambda) = -log(1 - lambda) - lambda (sub-exponential cumulant bound)
        psi_e = -np.log1p(-lam) - lam

        cum_lam = np.cumsum(lam)
        weighted_mean = np.cumsum(lam * u) / cum_lam
        radius = (log_term + np.cumsum(v_t * psi_e)) / cum_lam

        lo_u = np.clip(np.maximum.accumulate(weighted_mean - radius), 0.0, 1.0)
        hi_u = np.clip(np.minimum.accumulate(weighted_mean + radius), 0.0, 1.0)

        span = self.upper - self.lower
        return self.lower + span * lo_u, self.lower + span * hi_u

    def audit(
        self,
        values: np.ndarray,
        null_mean: float,
        alternative: str = "greater",
    ) -> AnytimeAuditResult:
        """Run an anytime-valid sequential hypothesis test and confidence sequence.

        Args:
            values: 1D array of observations or bounded causal influence scores.
            null_mean: Null hypothesis boundary ``mu_0`` strictly inside ``(lower, upper)``.
            alternative: Direction of test: ``'greater'`` (``H_1: mu > mu_0``),
                ``'less'`` (``H_1: mu < mu_0``), or ``'two-sided'`` (``H_1: mu != mu_0``).

        Returns:
            ``AnytimeAuditResult`` with e-value trajectory, anytime p-value, and CS bounds.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("values must be a non-empty 1D array.")
        u = self._rescale(arr)
        u0 = (float(null_mean) - self.lower) / (self.upper - self.lower)

        if alternative in ("greater", "less"):
            log_e = self._one_sided_log_e_vectorized(u, u0, direction=alternative)
        elif alternative == "two-sided":
            log_e_pos = self._one_sided_log_e_vectorized(u, u0, direction="greater")
            log_e_neg = self._one_sided_log_e_vectorized(u, u0, direction="less")
            # Equal 50/50 mixture of two nonnegative test supermartingales is a supermartingale
            m = np.maximum(log_e_pos, log_e_neg)
            log_e = m + np.log(0.5 * (np.exp(log_e_pos - m) + np.exp(log_e_neg - m)))
        else:
            raise ValueError(
                f"alternative must be 'greater', 'less', or 'two-sided', got {alternative!r}."
            )

        running_max_log_e = np.maximum.accumulate(log_e)
        threshold = np.log(1.0 / self.alpha)
        hits = np.where(running_max_log_e >= threshold)[0]
        stopping_time = int(hits[0] + 1) if hits.size > 0 else None

        final_log_e = float(log_e[-1])
        max_log_e = float(running_max_log_e[-1])
        log10_e = final_log_e / np.log(10.0)
        log10_max_e = max_log_e / np.log(10.0)
        e_val = float(np.exp(min(final_log_e, 700.0)))
        max_e_val = float(np.exp(min(max_log_e, 700.0)))
        anytime_p = float(min(1.0, np.exp(-max_log_e)))

        cs_low, cs_high = self.confidence_sequence(arr)
        return AnytimeAuditResult(
            e_value=e_val,
            max_e_value=max_e_val,
            log10_e_value=float(log10_e),
            log10_max_e_value=float(log10_max_e),
            anytime_p_value=anytime_p,
            rejected=bool(max_log_e >= threshold),
            stopping_time=stopping_time,
            sample_mean=float(arr.mean()),
            cs_low=cs_low,
            cs_high=cs_high,
            log_e_trajectory=log_e,
            null_mean=float(null_mean),
            alpha=self.alpha,
            alternative=alternative,
        )

    def batch_eprocess(
        self,
        matrix: np.ndarray,
        null_mean: float,
        alternative: str = "greater",
        device: str = "auto",
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Evaluate predictable betting e-processes in parallel across ``B`` streams.

        Uses PyTorch CUDA when available and requested, falling back cleanly to NumPy CPU.
        Produces float64 results identical to ``audit()`` within ``1e-12``.

        Args:
            matrix: 2D array of shape ``(n_streams, n_steps)`` in ``[lower, upper]``.
            null_mean: Null boundary strictly in ``(lower, upper)``.
            alternative: ``'greater'``, ``'less'``, or ``'two-sided'``.
            device: ``'auto'``, ``'cuda'``, ``'cuda:0'``, or ``'cpu'``.

        Returns:
            Tuple ``(final_log_e, max_log_e, resolved_device)`` where the arrays have shape
            ``(n_streams,)`` in natural log units.
        """
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] == 0:
            raise ValueError("matrix must be a 2D array with at least one step.")
        u = self._rescale(arr)
        u0 = (float(null_mean) - self.lower) / (self.upper - self.lower)
        resolved = _resolve_device(device)

        if resolved.startswith("cuda"):
            assert torch is not None
            dev = torch.device(resolved)
            u_t = torch.tensor(u, dtype=torch.float64, device=dev)

            def _torch_one_sided(direction: str) -> "torch.Tensor":
                x_t = (u_t - u0) if direction == "greater" else (u0 - u_t)
                lam_cap = (
                    self.max_bet_fraction / u0
                    if direction == "greater"
                    else self.max_bet_fraction / (1.0 - u0)
                )
                bsz, n = x_t.shape
                csum = torch.cumsum(x_t, dim=1)
                csq = torch.cumsum(x_t * x_t, dim=1)
                zeros = torch.zeros((bsz, 1), dtype=torch.float64, device=dev)
                prev_sum = torch.cat([zeros, csum[:, :-1]], dim=1)
                prev_sq = torch.cat([zeros, csq[:, :-1]], dim=1)
                prev_n = torch.arange(n, dtype=torch.float64, device=dev).unsqueeze(0)
                p0 = self.prior_strength
                mu_prev = prev_sum / (prev_n + p0)
                s2_prev = (prev_sq + 0.25 * p0) / (prev_n + p0)
                lam_t = torch.clamp(mu_prev / (s2_prev + 1e-8), 0.0, lam_cap)
                return torch.cumsum(torch.log1p(lam_t * x_t), dim=1)

            if alternative in ("greater", "less"):
                log_e_t = _torch_one_sided(alternative)
            elif alternative == "two-sided":
                lp = _torch_one_sided("greater")
                ln = _torch_one_sided("less")
                m = torch.maximum(lp, ln)
                log_e_t = m + torch.log(0.5 * (torch.exp(lp - m) + torch.exp(ln - m)))
            else:
                raise ValueError(f"Unsupported alternative: {alternative!r}.")
            final_log = log_e_t[:, -1].cpu().numpy()
            max_log = torch.amax(log_e_t, dim=1).cpu().numpy()
            return final_log, max_log, resolved

        if alternative in ("greater", "less"):
            log_e = self._one_sided_log_e_vectorized(u, u0, direction=alternative)
        elif alternative == "two-sided":
            lp = self._one_sided_log_e_vectorized(u, u0, direction="greater")
            ln = self._one_sided_log_e_vectorized(u, u0, direction="less")
            m = np.maximum(lp, ln)
            log_e = m + np.log(0.5 * (np.exp(lp - m) + np.exp(ln - m)))
        else:
            raise ValueError(f"Unsupported alternative: {alternative!r}.")
        return log_e[:, -1].copy(), np.max(log_e, axis=1).copy(), resolved
