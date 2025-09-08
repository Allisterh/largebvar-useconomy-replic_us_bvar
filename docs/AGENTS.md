# Repository Guidelines

## Project Structure & Modules
- **Core code:** `src/` (e.g., `var_model.py`, `irf_analysis.py`, `variance_decomposition.py`, `data_utils.py`, `diagnostics.py`, `seasonal_adjustment.py`, `coefficient_analysis.py`)
- **Entry points:** `main.py`, `example_usage.py`, `examples/` (e.g., `examples/proper_irf_analysis.py`)
- **Config:** `config/` (variables, plotting, Cholesky ordering)
- **Data:** `data/` (input CSVs) → read‑only in Git; avoid committing large/derived files
- **Outputs:** `plots/` (generated figures), `analysis_results_*` docs
- **Docs:** `docs/` (methodology, ordering guide)
- **Tests:** top‑level `test_*.py` and `tests/` package

## Build, Test, and Development
- **Create venv (Windows):** `python -m venv .venv && .\.venv\Scripts\Activate.ps1`
- **Install deps:** `pip install -r requirements.txt`
- **Run main:** `python main.py`
- **Run example IRF:** `python examples/proper_irf_analysis.py`
- **Run all tests:** `python -m pytest -q`

## Coding Style & Naming
- **Python style:** PEP 8, 4‑space indents, 88–100 char lines
- **Naming:** modules/files `snake_case.py`; functions `snake_case`; classes `PascalCase`
- **Docs:** module and public function docstrings summarize purpose, inputs, outputs
- **Imports:** standard → third‑party → local; prefer explicit over wildcard

## Testing Guidelines
- **Framework:** `pytest`
- **Locations:** `tests/` and top‑level `test_*.py`
- **Naming:** test files `test_<area>.py`; tests `test_<behavior>()`
- **Scope:** include data‑free unit tests for `src/` utilities; mark slow/IO as such (e.g., `@pytest.mark.slow`)
- **Run subset:** `pytest tests/test_transformations.py -q`

## Commit & Pull Requests
- **Commits:** imperative mood, concise: `fix: correct IRF ordering for AAA`
- **Link context:** reference issues (e.g., `Closes #12`), include before/after notes for model outputs
- **PRs must include:** goal summary, key changes, run commands used to validate, screenshots/figures (if plots change), and impact on `config/` or `data/`

## Security & Configuration
- **API keys:** set `FRED_API_KEY` via environment or `.env` (used by `fredapi`); never commit secrets
- **Data files:** store large datasets locally under `data/`; commit only small samples when necessary
- **Reproducibility:** record random seeds and configuration in PR descriptions when results change

## Tips for Contributors
- Prefer small, reviewable PRs
- Keep generated plots out of Git unless they illustrate a change
- Add minimal docs in `docs/` when introducing new methods or variables

