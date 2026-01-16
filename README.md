# Loss Modeling & Risk Analytics (Python)

This repo is a compact, end-to-end actuarial-style **frequency–severity** model that produces a simulated
**aggregate loss distribution** and common **tail risk** metrics.

The goal was to recreate the core logic behind how pricing / risk teams think about:
- **Frequency** (how often claims happen)
- **Severity** (how big claims are)
- **Aggregate loss** (total annual loss)
- **Tail risk** (VaR / TVaR)

## What’s inside
- `src/simulate.py` – Poisson frequency + Lognormal severity, Monte Carlo aggregate losses
- `src/risk_metrics.py` – VaR/TVaR and summary helpers
- `src/make_plots.py` – quick plots for distribution + tail
- `data/` – generated outputs (`aggregate_losses.csv`, `report.json`)
- `plots/` – saved charts

## Quick start
```bash
pip install -r requirements.txt
python -m src.simulate --n_sims 20000 --lambda_f 12 --mu 9.0 --sigma 1.0
python -m src.make_plots
```

## Notes
- Parameters are easy to swap to test different lines of business (higher frequency / heavier tail).
- Next improvements: fit parameters from real data, add reinsurance layers, add inflation/trend.
