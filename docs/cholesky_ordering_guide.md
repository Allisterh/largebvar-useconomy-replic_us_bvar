# Cholesky Ordering Guide for VAR Models

## Overview

This guide explains the proper use of Cholesky ordering in VAR models for macroeconomic analysis, based on industry standards and economic theory.

## Economic Theory Behind Ordering

The Cholesky decomposition requires a specific ordering of variables that reflects the causal structure of the economy. Variables ordered earlier are assumed to have contemporaneous effects on variables ordered later, but not vice versa.

### Standard Macroeconomic Ordering

```python
from config.cholesky_ordering import STANDARD_CHOLESKY_ORDERING

# Economic rationale:
# 1. Real Activity → affects everything contemporaneously
# 2. Labor → responds to real activity, affects prices and financial
# 3. Prices → respond to real activity and labor, affect financial
# 4. Financial → respond to all other sectors
```

## Usage

### Default Behavior (Recommended)

```python
from src.irf_analysis import analyze_irfs

# Uses standard economic ordering automatically
results = analyze_irfs(
    var_results, 
    periods=20,
    identification='cholesky',
    use_standard_ordering=True  # Default
)
```

### Custom Ordering

```python
# For specific research needs
custom_order = ['GDPC1', 'UNRATE', 'CPIAUCSL', 'DFF']

results = analyze_irfs(
    var_results,
    periods=20, 
    identification='cholesky',
    ordering=custom_order,
    use_standard_ordering=False
)
```

### Variable Categories

#### Real Activity (First)
- **GDPC1**: Real GDP
- **PCECC96**: Real consumption  
- **EXPGSC1**: Real exports
- **IMPGSC1**: Real imports
- **GCEC1**: Real government consumption
- **INDPRO**: Industrial production

#### Labor (Second)
- **PAYEMS**: Nonfarm payrolls
- **UNRATE**: Unemployment rate
- **TCU**: Capacity utilization

#### Prices (Third)
- **GDPDEF**: GDP deflator
- **PCEPI**: PCE price index
- **PCEPILFE**: Core PCE price index
- **CPIAUCSL**: Consumer price index
- **CPILFESL**: Core CPI
- **COMPRNFB**: Compensation per hour

#### Financial (Last)
- **DFF**: Federal funds rate
- **DGS2**, **DGS5**, **DGS10**: Treasury yields
- **AAA**, **BAA**: Corporate bond spreads

## Data Transformations

### Critical for Meaningful IRFs

```python
# For stationary IRF analysis, use first differences:
transform_config = {
    'GDPC1': 'first_difference',
    'UNRATE': 'first_difference', 
    'CPIAUCSL': 'first_difference',
    'DFF': 'first_difference'
}
```

### Expected Economic Responses

With proper ordering and transformations:
- ✅ **Unemployment decreases** after positive GDP shock
- ✅ **Prices increase** after positive demand shock  
- ✅ **Interest rates rise** after inflationary shock

## Validation Checklist

Before accepting IRF results:

1. **Economic Sensibility**
   - [ ] Unemployment responds negatively to GDP shocks
   - [ ] Prices respond positively to demand shocks
   - [ ] Interest rates respond to inflation

2. **Technical Requirements**
   - [ ] Variables are stationary (use first differences)
   - [ ] Ordering follows economic theory
   - [ ] Confidence intervals are reasonable

3. **Model Diagnostics**
   - [ ] Residuals are uncorrelated
   - [ ] Model is stable (eigenvalues < 1)
   - [ ] Lag length is appropriate

## Common Mistakes

### ❌ Alphabetical Ordering
```python
# Wrong - ignores economic structure
ordering = sorted(variables)  # ['AAA', 'BAA', 'CPIAUCSL', ...]
```

### ❌ Using Levels Data
```python
# Wrong - non-stationary data gives misleading IRFs
data = load_data()  # Raw levels
```

### ✅ Correct Approach
```python
# Right - economic ordering with stationary data
from config.cholesky_ordering import get_standard_ordering

ordering = get_standard_ordering(available_variables)
data = apply_transformations(raw_data, transform_config)
```

## References

- Christiano, L. J., Eichenbaum, M., & Evans, C. L. (1999). Monetary policy shocks: What have we learned and to what end?
- Sims, C. A. (1980). Macroeconomics and reality. Econometrica, 1-48.
- Crump, R. K., et al. (2024). Large Bayesian VARs with factor stochastic volatility.

## Implementation Files

- [`config/cholesky_ordering.py`](../config/cholesky_ordering.py): Standard ordering configuration
- [`src/irf_analysis.py`](../src/irf_analysis.py): IRF computation with proper ordering
- [`config/variables.py`](../config/variables.py): Variable transformation specifications

---
*Updated: September 2025*
*For questions, see the analysis examples in the `examples/` directory*