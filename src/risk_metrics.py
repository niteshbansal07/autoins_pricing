"""Risk metric helpers for aggregate loss modeling.

Contains:
  - var_tvar: Value-at-Risk and Tail Value-at-Risk for a simulated loss vector.
  - summarize: common descriptive stats.
"""
from __future__ import annotations
import numpy as np

def var_tvar(losses: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """Return (VaR_alpha, TVaR_alpha) for a 1D array of losses."""
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses must be a non-empty 1D array")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    var = float(np.quantile(losses, alpha, method="linear"))
    tail = losses[losses >= var]
    tvar = float(tail.mean()) if tail.size else var
    return var, tvar

def summarize(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "min": float(x.min()),
        "p50": float(np.quantile(x, 0.50, method="linear")),
        "p90": float(np.quantile(x, 0.90, method="linear")),
        "p95": float(np.quantile(x, 0.95, method="linear")),
        "p99": float(np.quantile(x, 0.99, method="linear")),
        "max": float(x.max()),
    }
