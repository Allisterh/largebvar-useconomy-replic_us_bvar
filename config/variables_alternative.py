"""
Alternative variable configuration aligned with the Crump et al. paper baseline.

High‑fidelity replication choices:
- Non‑stationary VAR: most series enter as 100×log levels; interest rates and
  unemployment enter in levels.
- Federal funds rate (DFF) omitted in the baseline (ZLB consideration in paper).
- Quarterly sample window: 1973:Q1–2019:Q1.
- Higher‑frequency series are averaged to quarterly before transformation.

Use by importing this module instead of `config.variables`, e.g.:
    from config.variables_alternative import FRED_VARIABLES as FRED_VARIABLES_PAPER
"""

# Paper‑aligned FRED series IDs and transformations for 31 variables
FRED_VARIABLES = {
    # === Real GDP and Components (log levels ×100) ===
    'GDPC1': {
        'name': 'Real Gross Domestic Product',
        'transform': 'log_level_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },
    'PCECC96': {
        'name': 'Real Personal Consumption Expenditures',
        'transform': 'log_level_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },
    'PRFIC1': {
        'name': 'Real Private Residential Fixed Investment',
        'transform': 'log_level_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },
    'PNFIC1': {
        'name': 'Real Private Nonresidential Fixed Investment',
        'transform': 'log_level_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },
    'EXPGSC1': {
        'name': 'Real Exports of Goods and Services',
        'transform': 'log_level_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },
    'IMPGSC1': {
        'name': 'Real Imports of Goods and Services',
        'transform': 'log_level_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },
    'GCEC1': {
        'name': 'Real Government Consumption Expenditures and Gross Investment',
        'transform': 'log_level_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },
    'FGCEC1': {
        'name': 'Real Federal Government Consumption Expenditures and Gross Investment',
        'transform': 'log_level_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },

    # === Prices and Wages (log levels ×100) ===
    'GDPDEF': {
        'name': 'Gross Domestic Product: Chain-type Price Index',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, 2012=100',
        'frequency': 'Q'
    },
    'PCEPI': {
        'name': 'Personal Consumption Expenditures: Chain-type Price Index',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, 2012=100',
        'frequency': 'Q'
    },
    'PCEPILFE': {
        'name': 'Core PCE Price Index (Excl. Food & Energy)',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, 2012=100',
        'frequency': 'Q'
    },
    'CPIAUCSL': {
        'name': 'CPI: All Items',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, 1982-84=100',
        'frequency': 'M'
    },
    'CPILFESL': {
        'name': 'CPI: All Items less Food & Energy',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, 1982-84=100',
        'frequency': 'M'
    },
    'COMPRNFB': {
        'name': 'Nonfarm Business: Compensation per Hour',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, Index 2012=100',
        'frequency': 'Q'
    },

    # === Labor, Activity, Sentiment ===
    'PAYEMS': {
        'name': 'All Employees: Total Nonfarm',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, Thous',
        'frequency': 'M'
    },
    'UNRATE': {
        'name': 'Unemployment Rate',
        'transform': 'level',
        'adjustment_status': 'sa',
        'units': 'SA, %',
        'frequency': 'M'
    },
    'INDPRO': {
        'name': 'Industrial Production Index',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, 2012=100',
        'frequency': 'M'
    },
    'TCU': {
        'name': 'Capacity Utilization: Manufacturing',
        'transform': 'level',
        'adjustment_status': 'sa',
        'units': 'SA, Percent of Capacity',
        'frequency': 'M'
    },
    'HOUST': {
        'name': 'Housing Starts',
        'transform': 'log_level_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Thous.Units',
        'frequency': 'M'
    },
    'DSPIC96': {
        'name': 'Real Disposable Personal Income',
        'transform': 'log_level_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'M'
    },
    'UMCSENT': {
        'name': 'Univ. of Michigan: Consumer Sentiment',
        'transform': 'level',
        'adjustment_status': 'nsa',
        'units': 'NSA, Q1-66=100',
        'frequency': 'M'
    },

    # === Interest Rates (levels; FFR omitted in baseline) ===
    'DGS2': {
        'name': '2-Year Treasury Constant Maturity Rate',
        'transform': 'level',
        'adjustment_status': 'raw',
        'units': '% p.a.',
        'frequency': 'D'
    },
    'DGS5': {
        'name': '5-Year Treasury Constant Maturity Rate',
        'transform': 'level',
        'adjustment_status': 'raw',
        'units': '% p.a.',
        'frequency': 'D'
    },
    'DGS10': {
        'name': '10-Year Treasury Constant Maturity Rate',
        'transform': 'level',
        'adjustment_status': 'raw',
        'units': '% p.a.',
        'frequency': 'D'
    },
    'AAA': {
        'name': "Moody's Seasoned Aaa Corporate Bond Yield",
        'transform': 'level',
        'adjustment_status': 'raw',
        'units': '% p.a.',
        'frequency': 'M'
    },
    'BAA': {
        'name': "Moody's Seasoned Baa Corporate Bond Yield",
        'transform': 'level',
        'adjustment_status': 'raw',
        'units': '% p.a.',
        'frequency': 'M'
    },

    # === Asset Prices and External (log levels ×100; VIX in levels) ===
    'DTWEXBGS': {
        'name': 'Nominal Broad U.S. Dollar Index',
        'transform': 'log_level_100',
        'adjustment_status': 'raw',
        'units': 'Mar-73=100',
        'frequency': 'D'
    },
    'SP500': {
        'name': 'S&P 500',
        'transform': 'log_level_100',
        'adjustment_status': 'raw',
        'units': '1941-43=10',
        'frequency': 'D'
    },
    'DCOILWTICO': {
        'name': 'Crude Oil Prices: WTI',
        'transform': 'log_level_100',
        'adjustment_status': 'raw',
        'units': '$/Barrel',
        'frequency': 'D'
    },
    'PNRGINDEXM': {
        'name': 'Global Price Index: Non‑Energy Commodities',
        'transform': 'log_level_100',
        'adjustment_status': 'raw',
        'units': 'AVG, Jan-02-70=100',
        'frequency': 'M'
    },
    'VIXCLS': {
        'name': 'CBOE Volatility Index: VIX',
        'transform': 'level',
        'adjustment_status': 'raw',
        'units': 'Index',
        'frequency': 'D'
    }
}

# Development subset without FFR (smaller, paper‑consistent)
DEV_FRED_VARIABLES = {
    'GDPC1': FRED_VARIABLES['GDPC1'],
    'CPIAUCSL': FRED_VARIABLES['CPIAUCSL'],
    'DGS10': FRED_VARIABLES['DGS10']
}

# Alternative series fallbacks (same as default config)
ALTERNATIVE_SERIES = {
    'DTWEXBGS': 'DTWEXM',           # If broad index not available, use major currencies
    'PNRGINDEXM': 'PALLFNFINDEXQ',  # Alternative commodity index
}

# Paper baseline sample window (1973:Q1–2019:Q1)
DATA_START = '1973-01-01'
DATA_END = '2019-03-31'

# Frequency lists for quarterly conversion logic
QUARTERLY_VARIABLES = [
    'GDPC1', 'PCECC96', 'PRFIC1', 'PNFIC1', 'EXPGSC1', 'IMPGSC1',
    'GCEC1', 'FGCEC1', 'GDPDEF', 'PCEPI', 'PCEPILFE', 'COMPRNFB'
]

MONTHLY_VARIABLES = [
    'CPIAUCSL', 'CPILFESL', 'PAYEMS', 'UNRATE', 'INDPRO', 'TCU',
    'HOUST', 'DSPIC96', 'UMCSENT', 'AAA', 'BAA', 'PNRGINDEXM'
]

DAILY_VARIABLES = [
    'DGS2', 'DGS5', 'DGS10', 'DTWEXBGS', 'SP500', 'DCOILWTICO', 'VIXCLS'
]

# Explicitly document excluded series for the paper baseline
EXCLUDED_SERIES = {
    'DFF': 'Federal funds effective rate omitted in paper baseline (ZLB considerations).'
}

