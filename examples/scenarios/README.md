# Scenario Constraints (Conditional Forecasts)

Provide a JSON mapping of horizons to variable name/value pairs. Horizons are 1-based.

Schema:

{
  "1": {"VAR_NAME": value, ...},
  "2": {"VAR_NAME": value}
}

Run:

`python main.py --step 8 --constraints examples/scenarios/financial_turmoil_2015q3.json --scenario-horizon 12`

Notes:
- Variable names must match model names printed in `output/var_model_summary.txt`.
- Values are in transformed units (e.g., log-level×100 for many real and price indices; levels for rates/unemployment).
- Scenario forecasts are saved to `output/scenario_forecast_*.csv`.

Canonical Scenarios
- Build canonical constraints (using unconditional median as baseline) via:

```
python scripts/scenarios/build_canonical_scenarios.py \
  --scenario financial_turmoil --horizon 12 \
  --output examples/scenarios/financial_turmoil_canonical.json

python scripts/scenarios/build_canonical_scenarios.py \
  --scenario no_supply_shock --horizon 12 \
  --output examples/scenarios/no_supply_shock_canonical.json

python scripts/scenarios/build_canonical_scenarios.py \
  --scenario tcja_baseline --horizon 12 \
  --output examples/scenarios/tcja_canonical.json
```

Then run step 8 with the generated JSON.
