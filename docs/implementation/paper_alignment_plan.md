# Paper Alignment Plan: Large Bayesian VAR (Crump et al.)

## Objectives

- Match the paper’s data construction (levels, seasonal adjustment, quarterly aggregation, sample window).
- Use Bayesian posterior uncertainty for IRFs/FEVD and center scenario analysis on conditional forecasts.
- Support canonical scenarios (financial turmoil, TCJA) and optional entropic tilting.
- Provide pathways for real‑time/vintage replication and documentation/tests to validate results.

## Phased Plan

### Phase 1 — Core Alignment
- Data transforms to log‑levels (×100) for most series; rates/unemployment in levels.
- Seasonal adjustment at native frequency before quarterly aggregation.
- Variable set parity and sample window consistency (1973:Q1–2019:Q1).
- Correct generalized IRFs; replace bootstrap with posterior credible intervals for IRF/FEVD.

### Phase 2 — Scenario Engine
- Expose conditional forecasts via CLI and config files.
- Add canonical scenarios (financial turmoil, TCJA) with example configs and scripts.
- Implement entropic tilting for SEP/anchoring.

### Phase 3 — Real‑Time & Evaluation
- ALFRED vintages integration and SPF/Greenbook comparisons (where feasible).
- Validation scripts and reproducible outputs.

### Phase 4 — Docs & Testing
- Examples, CLI usage, acceptance tests, and replication notes.

## Checklist (with File Pointers)

### Data Construction
- [ ] Transform design
  - [ ] Introduce explicit transform keys: `log_level_100`, `rate_level`, `log_diff_100` (only as needed).
  - [ ] Update per‑series mappings to match Table 1.
  - Files: `src/data_downloader.py`, `config/variables.py`.
- [ ] Seasonal adjustment and aggregation
  - [ ] Adjust NSA series at native frequency (X‑13/STL), then aggregate to quarterly.
  - Files: `src/data_downloader.py`, `src/seasonal_adjustment.py`.
- [ ] Variable set and sample window
  - [ ] Align to the paper’s 31 variables; remove extras unless explicitly included.
  - [ ] Enforce 1973:Q1–2019:Q1 window.
  - [ ] Use `ALTERNATIVE_SERIES` when primaries are missing.
  - Files: `config/variables.py`, `src/data_downloader.py`.
- [ ] Missing data policy and logging
  - [ ] Report excluded variables and reasons instead of silent drops (>50% missing).
  - Files: `src/var_model.py`.

### Identification and IRFs/FEVD
- [ ] Correct GIRF implementation (order invariant, uses full Σ)
  - Files: `src/irf_analysis.py`.
- [ ] Posterior credible intervals for IRFs
  - [ ] Use `LargeBayesianVAR.get_irf_summary()` (68% & 90% bands) in the pipeline.
  - Files: `src/irf_analysis.py`, `src/var_model.py` (ensure compatibility).
- [ ] Posterior FEVD summaries
  - [ ] Use `LargeBayesianVAR.get_fevd_summary()` and document in outputs.
  - Files: `src/variance_decomposition.py`.

### Conditional Forecasting (Scenarios)
- [ ] CLI exposure for scenarios
  - [ ] Add `--scenario` and `--constraints` (JSON/YAML) to drive conditional forecasts.
  - Files: `main.py`, `src/bayesian/core_bayesian_var.py`, `src/analysis/conditional_forecasting.py`.
- [ ] Canonical scenarios
  - [ ] Financial turmoil (2015:Q3) and TCJA example configs and scripts.
  - Files: `scripts/replicate_paper.py`, `docs/` examples.

### Entropic Tilting (Anchoring)
- [ ] Implement entropic tilting
  - [ ] Reweight posterior draws to match external moments (SEP medians, targets).
  - [ ] CLI: `--tilt-targets path` to specify targets and horizons.
  - Files: `src/analysis/entropic_tilting.py` (new), `main.py`.

### Real‑Time/Vintages and Evaluation
- [ ] ALFRED integration
  - [ ] Fetch series by vintage; CLI: `--vintage YYYY-MM-DD`.
  - Files: `src/alfred.py` or `src/data_utils.py` additions.
- [ ] Evaluation scripts
  - [ ] Error distributions vs SPF/Greenbook; real‑time fan charts.
  - Files: `scripts/evaluate_realtime.py`, `scripts/compare_spf_greenbook.py`.

### Diagnostics, Tests, Docs
- [ ] Tests
  - [ ] Transforms, GIRF, posterior IRFs/FEVD, conditional paths, tilting.
  - Files: `tests/unit/…`, `tests/integration/…`.
- [ ] Docs
  - [ ] Transform table (series → transform), scenario schemas, replication steps.
  - Files: `README.md`, `docs/` additions.
- [ ] Performance defaults
  - [ ] Sensible defaults for `n_draws`/`n_burnin`; `--fast-mode` maintained.

## Acceptance Criteria

- Data transforms
  - Level series: `100*log(X_t)`; rates/unemployment in levels; seasonal adjustment at native frequency; quarterly aggregation documented per series.
  - Window equals 1973:Q1–2019:Q1; variable list matches paper; logs show fallbacks and exclusions.
- GIRF correctness
  - Impact at h=0: GIRF(j,i) = Σ_{ji}/sqrt(Σ_{ii}); order invariance verified.
- Posterior IRFs/FEVD
  - Plots show median with 68%/90% credible bands; CSVs include intervals.
- Conditional scenarios
  - CLI executes constraints from JSON/YAML; financial turmoil and TCJA reproduce expected qualitative responses.
- Entropic tilting
  - Reweighted forecasts match specified medians within tolerance; diagnostics saved.
- Real‑time
  - Given a vintage date, model estimates on vintage data; evaluation scripts produce RMSE/coverage stats.

## Risks & Notes

- Transform table fidelity is critical; encode Table 1 as machine‑readable mapping.
- Posterior run time for 31 variables can be heavy; keep `--fast-mode` and tune draws.
- X‑13 availability varies; STL fallback required with clear warnings.

## Suggested Milestones

- M1 (Core, 1–2 days): Transform alignment + seasonal flow + variable set/window; baseline BVAR runs.
- M2 (IRFs, 1–2 days): GIRF fixed; posterior IRF/FEVD bands wired; plots updated.
- M3 (Scenarios, 2–3 days): CLI scenarios + two canonical configs + outputs documented.
- M4 (Tilting, 2–3 days): Entropic tilting module + anchoring/SEP example.
- M5 (Real‑time, 3–5 days): ALFRED adapter + evaluation scripts + docs.
- M6 (Docs/Tests, 1–2 days): Tests and replication guide finalized.

