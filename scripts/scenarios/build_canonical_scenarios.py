#!/usr/bin/env python3
"""
Build canonical scenario constraint JSONs aligned with the paper's narratives.

Scenarios:
  - financial_turmoil: shock key financial variables (stock, VIX, oil, USD, yields)
  - no_supply_shock: as above but hold oil/commodities at unconditional paths
  - tcja_baseline (proxy): small positive path for GDP (and slightly for core prices)

This script uses the estimated BVAR to compute unconditional median forecasts and
creates constraints as deviations in the transformed space.
"""

import os
import json
import argparse
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from main import VARPipeline
from config.variables import FRED_VARIABLES


def pct_to_log100_delta(pct_change: float) -> float:
    """Convert a percent change (e.g., -0.05 for -5%) to delta in 100*log(X)."""
    return 100.0 * np.log(1.0 + pct_change)


def build_financial_turmoil(uncond_median, names):
    """Quarter-1 shocks to key financial variables to mimic 2015:Q3 turmoil."""
    name_to_idx = {n: i for i, n in enumerate(names)}
    H = uncond_median.shape[0]
    constraints = {}

    # Horizon 1 shocks (illustrative magnitudes)
    h = 1
    c = {}
    if 'SP500' in name_to_idx:  # -5% stock
        idx = name_to_idx['SP500']
        c['SP500'] = float(uncond_median[h-1, idx] + pct_to_log100_delta(-0.05))
    if 'VIXCLS' in name_to_idx:  # +10 index points
        idx = name_to_idx['VIXCLS']
        c['VIXCLS'] = float(uncond_median[h-1, idx] + 10.0)
    if 'DCOILWTICO' in name_to_idx:  # -10% oil
        idx = name_to_idx['DCOILWTICO']
        c['DCOILWTICO'] = float(uncond_median[h-1, idx] + pct_to_log100_delta(-0.10))
    if 'DTWEXBGS' in name_to_idx:  # +2% USD
        idx = name_to_idx['DTWEXBGS']
        c['DTWEXBGS'] = float(uncond_median[h-1, idx] + pct_to_log100_delta(0.02))
    if 'DGS10' in name_to_idx:  # -20 bps 10y yield
        idx = name_to_idx['DGS10']
        c['DGS10'] = float(uncond_median[h-1, idx] - 0.20)
    if 'AAA' in name_to_idx:  # +10 bps Aaa spread
        idx = name_to_idx['AAA']
        c['AAA'] = float(uncond_median[h-1, idx] + 0.10)
    if 'BAA' in name_to_idx:  # +15 bps Baa spread
        idx = name_to_idx['BAA']
        c['BAA'] = float(uncond_median[h-1, idx] + 0.15)

    if c:
        constraints[str(h)] = c

    # Optional horizon 2 slight persistence on yields/spreads
    h = 2
    c2 = {}
    if 'DGS10' in name_to_idx:
        idx = name_to_idx['DGS10']
        c2['DGS10'] = float(uncond_median[h-1, idx] - 0.10)
    if 'AAA' in name_to_idx:
        idx = name_to_idx['AAA']
        c2['AAA'] = float(uncond_median[h-1, idx] + 0.05)
    if 'BAA' in name_to_idx:
        idx = name_to_idx['BAA']
        c2['BAA'] = float(uncond_median[h-1, idx] + 0.10)
    if c2:
        constraints[str(h)] = c2

    return constraints


def build_no_supply_shock(uncond_median, names):
    """
    Demand-driven financial tightening while holding supply-sensitive prices at unconditional.
    """
    base = build_financial_turmoil(uncond_median, names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    h = 1
    if str(h) not in base:
        base[str(h)] = {}
    c = base[str(h)]

    # Override oil and non-energy commodities to unconditional
    for v in ('DCOILWTICO', 'PNRGINDEXM'):
        if v in name_to_idx:
            idx = name_to_idx[v]
            c[v] = float(uncond_median[h-1, idx])
    base[str(h)] = c
    return base


def build_tcja_proxy(uncond_median, names):
    """Proxy TCJA scenario with small positive GDP and slight core price lift."""
    name_to_idx = {n: i for i, n in enumerate(names)}
    constraints = {}
    # GDP path +0.3%, +0.4%, +0.4% (log-level*100 deltas)
    if 'GDPC1' in name_to_idx:
        for h, pct in [(1, 0.003), (2, 0.004), (3, 0.004)]:
            idx = name_to_idx['GDPC1']
            constraints.setdefault(str(h), {})['GDPC1'] = float(uncond_median[h-1, idx] + pct_to_log100_delta(pct))
    # Core PCE slight lift at h=1 (e.g., +0.05%)
    if 'PCEPILFE' in name_to_idx:
        h = 1
        idx = name_to_idx['PCEPILFE']
        constraints.setdefault(str(h), {})['PCEPILFE'] = float(uncond_median[h-1, idx] + pct_to_log100_delta(0.0005))
    return constraints


def main():
    ap = argparse.ArgumentParser(description='Build canonical scenario constraints')
    ap.add_argument('--scenario', required=True, choices=['financial_turmoil', 'no_supply_shock', 'tcja_baseline'])
    ap.add_argument('--horizon', type=int, default=12)
    ap.add_argument('--output', required=True, help='Output JSON path')
    ap.add_argument('--max-lags', type=int, default=8)
    ap.add_argument('--gibbs-draws', type=int, default=500)
    args = ap.parse_args()

    # Build/estimate a quick model to get unconditional forecast medians
    pipeline = VARPipeline()
    # Load or download data (assumes data already downloaded via main step 1)
    pipeline.step1_download_data(force_download=False, use_crump_transformations=True, fred_variables=FRED_VARIABLES)
    pipeline.step2_estimate_var(max_lags=args.max_lags, n_draws=args.gibbs_draws)

    # Unconditional forecast draws via conditional_forecast with empty constraints
    draws = pipeline.var_model.results.get_conditional_forecast(horizon=args.horizon, constraints={})
    median = np.nanmedian(draws, axis=0)  # (horizon, n_vars)
    names = list(pipeline.var_model.results.names)

    if args.scenario == 'financial_turmoil':
        constraints = build_financial_turmoil(median, names)
    elif args.scenario == 'no_supply_shock':
        constraints = build_no_supply_shock(median, names)
    else:
        constraints = build_tcja_proxy(median, names)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(constraints, f, indent=2)
    print(f"Saved constraints to {args.output}")


if __name__ == '__main__':
    main()

