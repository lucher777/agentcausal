"""Counterfactual guardrail mediation decomposition and safety-utility Pareto frontier.

Implements cross-fitted multiply-robust efficient influence function (EIF) estimation
for path-specific causal mediation (Tchetgen Tchetgen & Shpitser, 2012; Farbmacher
et al., 2022), separating Natural Direct Effects (NDE) from Natural Indirect Effects
(NIE, e.g., over-refusal or reasoning attenuation) and tracing doubly-robust
counterfactual safety-utility Pareto frontiers across guardrail trigger policies.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

from .types import GuardrailMediationResult, ParetoPoint


def _expand_features(x: np.ndarray, m: Optional[np.ndarray] = None) -> np.ndarray:
    """Create quadratic and interaction basis features for flexible nuisance fits."""
    x2 = np.asarray(x, dtype=np.float64)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)
    base = np.hstack([x2, x2 * x2])
    if m is None:
        return base
    mv = np.asarray(m, dtype=np.float64).reshape(-1, 1)
    return np.hstack([base, mv, mv * x2])


class GuardrailDecomposer:
    """Multiply-robust path-specific causal decomposer for LLM guardrails.

    Decomposes the total causal effect of deploying a guardrail (``A in {0, 1}``) on a
    target outcome ``Y`` into:

    - **Natural Direct Effect (NDE)**: ``E[Y(1, M(0)) - Y(0, M(0))]``, capturing the
      direct safety/quality effect holding the refusal/mediation pathway ``M`` fixed at
      its baseline counterfactual ``M(0)``.
    - **Natural Indirect Effect (NIE)**: ``E[Y(1, M(1)) - Y(1, M(0))]``, capturing the
      indirect utility cost or safety gain mediated through intermediate refusal or
      reasoning shifts ``M``.

    Uses cross-fitted Bayes-rule density-ratio efficient influence functions that work
    uniformly for binary or continuous mediators ``M`` and satisfy the exact identity
    ``NDE + NIE == ATE`` to machine precision.

    Args:
        n_folds: Number of cross-fitting folds ``K >= 2``.
        propensity_clip: Clipping margin for treatment propensity scores.
        ridge_alpha: L2 penalty for outcome regression models.
        alpha: Two-sided confidence interval significance level.
        random_state: Random seed for K-fold splitting.
    """

    def __init__(
        self,
        n_folds: int = 5,
        propensity_clip: float = 0.02,
        ridge_alpha: float = 1.0,
        alpha: float = 0.05,
        random_state: int = 42,
    ) -> None:
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}.")
        if not (0.0 < propensity_clip < 0.25):
            raise ValueError(f"propensity_clip must lie in (0, 0.25), got {propensity_clip}.")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
        self.n_folds = int(n_folds)
        self.propensity_clip = float(propensity_clip)
        self.ridge_alpha = float(ridge_alpha)
        self.alpha = float(alpha)
        self.random_state = int(random_state)

    def decompose(
        self,
        contexts: np.ndarray,
        guardrail: np.ndarray,
        mediator: np.ndarray,
        outcome: np.ndarray,
    ) -> GuardrailMediationResult:
        """Estimate ATE, NDE, and NIE via cross-fitted efficient influence functions.

        Args:
            contexts: Pre-treatment query/conversation covariates ``X`` of shape ``(n, d)``.
            guardrail: Binary guardrail activation indicator ``A in {0, 1}`` of shape ``(n,)``.
            mediator: Intermediate refusal or reasoning-shift variable ``M`` of shape ``(n,)``.
            outcome: Scalar downstream outcome ``Y`` (utility or safety) of shape ``(n,)``.

        Returns:
            ``GuardrailMediationResult`` with exact EIF point estimates, SEs, and CIs.
        """
        x = np.asarray(contexts, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        a = np.asarray(guardrail, dtype=np.int64).reshape(-1)
        m = np.asarray(mediator, dtype=np.float64).reshape(-1)
        y = np.asarray(outcome, dtype=np.float64).reshape(-1)

        n = x.shape[0]
        if not (a.shape[0] == m.shape[0] == y.shape[0] == n):
            raise ValueError("contexts, guardrail, mediator, and outcome must share length n.")
        if set(np.unique(a)).difference({0, 1}):
            raise ValueError("guardrail must be binary with values in {0, 1}.")

        eps = self.propensity_clip
        psi_11 = np.empty(n, dtype=np.float64)
        psi_00 = np.empty(n, dtype=np.float64)
        psi_10 = np.empty(n, dtype=np.float64)

        is_binary_m = bool(np.all(np.isin(np.unique(m), [0.0, 1.0])))
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for tr, va in kf.split(np.arange(n)):
            x_tr, x_va = x[tr], x[va]
            a_tr, a_va = a[tr], a[va]
            m_tr, m_va = m[tr], m[va]
            y_tr, y_va = y[tr], y[va]

            fx_tr = _expand_features(x_tr)
            fx_va = _expand_features(x_va)
            fxm_tr = _expand_features(x_tr, m_tr)
            fxm_va = _expand_features(x_va, m_va)

            # 1. Treatment propensities e1(X) = P(A=1|X) and e1(X,M) = P(A=1|X,M)
            clf_x = LogisticRegression(C=1.0, max_iter=500).fit(fx_tr, a_tr)
            e1_x_va = np.clip(clf_x.predict_proba(fx_va)[:, 1], eps, 1.0 - eps)

            clf_xm = LogisticRegression(C=1.0, max_iter=500).fit(fxm_tr, a_tr)
            e1_xm_va = np.clip(clf_xm.predict_proba(fxm_va)[:, 1], eps, 1.0 - eps)

            # 2. Marginal outcome models nu_1(X) = E[Y|A=1,X], nu_0(X) = E[Y|A=0,X]
            m1_tr = a_tr == 1
            m0_tr = a_tr == 0
            reg_nu1 = Ridge(alpha=self.ridge_alpha).fit(fx_tr[m1_tr], y_tr[m1_tr])
            reg_nu0 = Ridge(alpha=self.ridge_alpha).fit(fx_tr[m0_tr], y_tr[m0_tr])
            nu1_va = reg_nu1.predict(fx_va)
            nu0_va = reg_nu0.predict(fx_va)

            # 3. Conditional outcome model mu(1, M, X) = E[Y | A=1, M, X]
            reg_mu1 = Ridge(alpha=self.ridge_alpha).fit(fxm_tr[m1_tr], y_tr[m1_tr])
            mu1_va = reg_mu1.predict(fxm_va)

            # 4. Nested conditional expectation eta_{1,0}(X) = E[mu(1, M, X) | A=0, X]
            if is_binary_m and np.unique(m_tr[m0_tr]).size == 2:
                clf_m0 = LogisticRegression(C=1.0, max_iter=500).fit(
                    fx_tr[m0_tr], m_tr[m0_tr].astype(np.int64)
                )
                pm1_a0_va = np.clip(clf_m0.predict_proba(fx_va)[:, 1], 1e-6, 1.0 - 1e-6)
                mu1_m1_va = reg_mu1.predict(_expand_features(x_va, np.ones(va.size)))
                mu1_m0_va = reg_mu1.predict(_expand_features(x_va, np.zeros(va.size)))
                eta10_va = pm1_a0_va * mu1_m1_va + (1.0 - pm1_a0_va) * mu1_m0_va
            else:
                mu1_on_tr0 = reg_mu1.predict(fxm_tr[m0_tr])
                reg_eta = Ridge(alpha=self.ridge_alpha).fit(fx_tr[m0_tr], mu1_on_tr0)
                eta10_va = reg_eta.predict(fx_va)

            # 5. Assemble efficient influence functions
            ind1 = (a_va == 1).astype(np.float64)
            ind0 = (a_va == 0).astype(np.float64)

            psi_11[va] = nu1_va + (ind1 / e1_x_va) * (y_va - nu1_va)
            psi_00[va] = nu0_va + (ind0 / (1.0 - e1_x_va)) * (y_va - nu0_va)

            # Bayes-rule mediator density ratio weight w = ((1 - e1(X,M)) / e1(X,M)) / (1 - e1(X))
            w_med = ((1.0 - e1_xm_va) / e1_xm_va) / (1.0 - e1_x_va)
            psi_10[va] = (
                eta10_va
                + (ind0 / (1.0 - e1_x_va)) * (mu1_va - eta10_va)
                + ind1 * w_med * (y_va - mu1_va)
            )

        eif_ate = psi_11 - psi_00
        eif_nde = psi_10 - psi_00
        eif_nie = psi_11 - psi_10

        ate = float(eif_ate.mean())
        nde = float(eif_nde.mean())
        nie = float(eif_nie.mean())

        ate_se = float(eif_ate.std(ddof=1) / np.sqrt(n))
        nde_se = float(eif_nde.std(ddof=1) / np.sqrt(n))
        nie_se = float(eif_nie.std(ddof=1) / np.sqrt(n))
        z = float(norm.ppf(1.0 - self.alpha / 2.0))

        return GuardrailMediationResult(
            ate=ate,
            nde=nde,
            nie=nie,
            ate_se=ate_se,
            nde_se=nde_se,
            nie_se=nie_se,
            ate_ci=(ate - z * ate_se, ate + z * ate_se),
            nde_ci=(nde - z * nde_se, nde + z * nde_se),
            nie_ci=(nie - z * nie_se, nie + z * nie_se),
            additivity_error=float(abs((nde + nie) - ate)),
            eif_ate=eif_ate,
            eif_nde=eif_nde,
            eif_nie=eif_nie,
            alpha=self.alpha,
        )

    def pareto_frontier(
        self,
        contexts: np.ndarray,
        guardrail: np.ndarray,
        safety_outcome: np.ndarray,
        utility_outcome: np.ndarray,
        risk_scores: np.ndarray,
        thresholds: Sequence[float],
    ) -> List[ParetoPoint]:
        """Trace the counterfactual safety-utility Pareto frontier across risk thresholds.

        For each candidate threshold ``tau``, defines the deterministic counterfactual
        guardrail policy ``pi_tau(X) = I(risk_score(X) >= tau)`` and estimates both
        counterfactual safety ``E[S(pi_tau)]`` and counterfactual task utility
        ``E[U(pi_tau)]`` via cross-fitted Augmented Inverse Propensity Weighting (AIPW).

        Args:
            contexts: Pre-treatment covariates ``X`` of shape ``(n, d)``.
            guardrail: Observed binary guardrail indicator ``A in {0, 1}`` of shape ``(n,)``.
            safety_outcome: Observed safety score ``S`` of shape ``(n,)``.
            utility_outcome: Observed utility score ``U`` of shape ``(n,)``.
            risk_scores: Continuous risk classifier scores of shape ``(n,)``.
            thresholds: Sequence of candidate trigger thresholds ``tau``.

        Returns:
            List of ``ParetoPoint`` instances annotated with ``is_pareto_optimal``.
        """
        x = np.asarray(contexts, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        a = np.asarray(guardrail, dtype=np.int64).reshape(-1)
        s_out = np.asarray(safety_outcome, dtype=np.float64).reshape(-1)
        u_out = np.asarray(utility_outcome, dtype=np.float64).reshape(-1)
        scores = np.asarray(risk_scores, dtype=np.float64).reshape(-1)
        n = x.shape[0]
        eps = self.propensity_clip

        # Cross-fit nuisance functions once, then evaluate all policy thresholds exactly
        e1_hat = np.empty(n, dtype=np.float64)
        mu_s1 = np.empty(n, dtype=np.float64)
        mu_s0 = np.empty(n, dtype=np.float64)
        mu_u1 = np.empty(n, dtype=np.float64)
        mu_u0 = np.empty(n, dtype=np.float64)

        x_aug = np.column_stack([x, scores])
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for tr, va in kf.split(np.arange(n)):
            fx_tr = _expand_features(x_aug[tr])
            fx_va = _expand_features(x_aug[va])
            a_tr = a[tr]
            m1, m0 = a_tr == 1, a_tr == 0

            clf = LogisticRegression(C=1.0, max_iter=500).fit(fx_tr, a_tr)
            e1_hat[va] = np.clip(clf.predict_proba(fx_va)[:, 1], eps, 1.0 - eps)

            mu_s1[va] = Ridge(alpha=self.ridge_alpha).fit(fx_tr[m1], s_out[tr][m1]).predict(fx_va)
            mu_s0[va] = Ridge(alpha=self.ridge_alpha).fit(fx_tr[m0], s_out[tr][m0]).predict(fx_va)
            mu_u1[va] = Ridge(alpha=self.ridge_alpha).fit(fx_tr[m1], u_out[tr][m1]).predict(fx_va)
            mu_u0[va] = Ridge(alpha=self.ridge_alpha).fit(fx_tr[m0], u_out[tr][m0]).predict(fx_va)

        # Potential outcome AIPW scores for A=1 and A=0
        psi_s1 = mu_s1 + ((a == 1) / e1_hat) * (s_out - mu_s1)
        psi_s0 = mu_s0 + ((a == 0) / (1.0 - e1_hat)) * (s_out - mu_s0)
        psi_u1 = mu_u1 + ((a == 1) / e1_hat) * (u_out - mu_u1)
        psi_u0 = mu_u0 + ((a == 0) / (1.0 - e1_hat)) * (u_out - mu_u0)

        points: List[ParetoPoint] = []
        for tau in thresholds:
            pi = (scores >= float(tau)).astype(np.float64)
            s_pi = pi * psi_s1 + (1.0 - pi) * psi_s0
            u_pi = pi * psi_u1 + (1.0 - pi) * psi_u0
            points.append(
                ParetoPoint(
                    threshold=float(tau),
                    safety_rate=float(s_pi.mean()),
                    safety_se=float(s_pi.std(ddof=1) / np.sqrt(n)),
                    utility_rate=float(u_pi.mean()),
                    utility_se=float(u_pi.std(ddof=1) / np.sqrt(n)),
                    trigger_rate=float(pi.mean()),
                )
            )

        # Identify non-dominated Pareto frontier maximizing (safety_rate, utility_rate)
        for i, p_i in enumerate(points):
            dominated = False
            for j, p_j in enumerate(points):
                if i == j:
                    continue
                if (
                    p_j.safety_rate >= p_i.safety_rate
                    and p_j.utility_rate >= p_i.utility_rate
                    and (p_j.safety_rate > p_i.safety_rate or p_j.utility_rate > p_i.utility_rate)
                ):
                    dominated = True
                    break
            p_i.is_pareto_optimal = not dominated

        return points
