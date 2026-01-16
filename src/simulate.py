"""Frequency–Severity aggregate loss simulation (Poisson–Lognormal).

Why this matters:
  - Pricing: expected loss + risk load depends on the loss distribution tail.
  - Capital: VaR/TVaR are common tail measures used in risk management.

Run:
  python -m src.simulate --n_sims 20000 --lambda_f 12 --mu 9.0 --sigma 1.0
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .risk_metrics import var_tvar, summarize

def simulate_aggregate_losses(
    n_sims: int,
    lambda_f: float,
    sev_mu: float,
    sev_sigma: float,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Frequency: Poisson
    n_claims = rng.poisson(lam=lambda_f, size=n_sims)

    # Severity: Lognormal (parameterized by underlying normal mu/sigma)
    # For each simulation, sample n_claims severities and sum.
    losses = np.zeros(n_sims, dtype=float)
    for i, k in enumerate(n_claims):
        if k > 0:
            severities = rng.lognormal(mean=sev_mu, sigma=sev_sigma, size=k)
            losses[i] = severities.sum()
    return losses

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sims", type=int, default=20000)
    ap.add_argument("--lambda_f", type=float, default=12.0)
    ap.add_argument("--mu", type=float, default=9.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="data")
    args = ap.parse_args()

    losses = simulate_aggregate_losses(
        n_sims=args.n_sims,
        lambda_f=args.lambda_f,
        sev_mu=args.mu,
        sev_sigma=args.sigma,
        seed=args.seed,
    )

    stats = summarize(losses)
    var95, tvar95 = var_tvar(losses, 0.95)
    var99, tvar99 = var_tvar(losses, 0.99)

    report = {
        "inputs": {
            "n_sims": args.n_sims,
            "lambda_frequency": args.lambda_f,
            "severity_lognormal_mu": args.mu,
            "severity_lognormal_sigma": args.sigma,
            "seed": args.seed,
        },
        "summary": stats,
        "risk_metrics": {
            "VaR_95": var95,
            "TVaR_95": tvar95,
            "VaR_99": var99,
            "TVaR_99": tvar99,
        },
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.Series(losses, name="aggregate_loss").to_csv(outdir/"aggregate_losses.csv", index=False)
    (outdir/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
