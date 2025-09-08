"""
Entropic Tilting for Posterior Predictive Forecasts

Implements KL-minimizing reweighting of posterior forecast draws to match
specified moment targets (currently: mean constraints).
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize


def _extract_g(draws: np.ndarray, var_names: List[str], constraints: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the constraint matrix G and target vector t.

    draws: (n_draws, horizon, n_vars)
    constraints: list of {"h": int, "var": str, "target": float}
    Returns (G, t) where G is (n_draws, k) and t is (k,)
    """
    n_draws, horizon, n_vars = draws.shape
    k = len(constraints)
    G = np.zeros((n_draws, k))
    t = np.zeros(k)

    name_to_idx = {n: i for i, n in enumerate(var_names)}

    for j, c in enumerate(constraints):
        h = int(c["h"])  # 1-based horizon
        v = c["var"]
        tgt = float(c["target"])
        if v not in name_to_idx:
            raise ValueError(f"Unknown variable in tilting constraints: {v}")
        vi = name_to_idx[v]
        if h < 1 or h > horizon:
            raise ValueError(f"Tilting horizon {h} out of range (1..{horizon})")
        # draw j, horizon h-1, variable vi
        G[:, j] = draws[:, h - 1, vi]
        t[j] = tgt

    return G, t


def entropic_tilt_mean(draws: np.ndarray, var_names: List[str], constraints: List[Dict]) -> np.ndarray:
    """
    Compute tilting weights to match mean constraints.

    Minimizes: phi(lambda) = log( sum_i exp(lambda' g_i) / n ) - lambda' t

    Returns weights w of shape (n_draws,), sum(w)=1, w>=0.
    """
    n_draws = draws.shape[0]
    G, t = _extract_g(draws, var_names, constraints)

    def dual_obj(lmbda: np.ndarray) -> float:
        z = G @ lmbda  # (n_draws,)
        m = np.max(z)
        # log-sum-exp for numerical stability
        lse = m + np.log(np.sum(np.exp(z - m))) - np.log(n_draws)
        return float(lse - lmbda @ t)

    def dual_grad(lmbda: np.ndarray) -> np.ndarray:
        z = G @ lmbda
        m = np.max(z)
        exp_z = np.exp(z - m)
        denom = np.sum(exp_z)
        w = exp_z / denom  # unnormalized; normalized by denom
        # gradient: E_w[g] - t
        Ew_g = G.T @ w / 1.0  # since w sums to 1 after /denom
        return Ew_g - t

    res = minimize(
        fun=dual_obj,
        x0=np.zeros(G.shape[1]),
        jac=dual_grad,
        method='BFGS',
        options={'gtol': 1e-8, 'maxiter': 1000}
    )
    if not res.success:
        raise RuntimeError(f"Entropic tilting optimization failed: {res.message}")

    lmbda = res.x
    z = G @ lmbda
    m = np.max(z)
    w = np.exp(z - m)
    w = w / np.sum(w)
    return w


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: List[float]) -> np.ndarray:
    """
    Compute weighted quantiles for a 1D array.
    """
    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]
    cdf = np.cumsum(w)
    out = np.interp(quantiles, cdf, v)
    return out


def summarize_tilted(draws: np.ndarray, weights: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Summarize weighted predictive draws with mean, 5/50/95th percentiles per (horizon, var).
    Returns dict with keys: 'mean', 'p5', 'p50', 'p95' arrays shaped (horizon, n_vars).
    """
    n_draws, horizon, n_vars = draws.shape
    mean = np.tensordot(weights, draws, axes=(0, 0))  # (horizon, n_vars)
    p5 = np.zeros((horizon, n_vars))
    p50 = np.zeros((horizon, n_vars))
    p95 = np.zeros((horizon, n_vars))
    for h in range(horizon):
        for v in range(n_vars):
            vals = draws[:, h, v]
            q = weighted_quantiles(vals, weights, [0.05, 0.5, 0.95])
            p5[h, v], p50[h, v], p95[h, v] = q
    return {'mean': mean, 'p5': p5, 'p50': p50, 'p95': p95}

