"""
Deposit-focused variable configuration that extends the paper-aligned baseline.

Purpose:
- Add banking deposit aggregates (total, checkable, time, small/large bank splits)
- Add financial conditions and uncertainty indexes (ANFCI/KCFSI, EPU)
- Keep the paper baseline macro block for comparability

Notes on transformations:
- Baseline (paper-style) uses non-stationary levels for most real and price series
  via 100×log levels ("log_level_100"). Interest rates and unemployment are in
  levels. This module follows that convention for added deposit series. If you
  prefer stationarity, change deposit transforms to "log_diff_100".
- Weekly series (e.g., ANFCI) are marked here with frequency 'D' so the current
  downloader converts them to quarterly via mean aggregation. If you later add
  native weekly handling, switch frequency to 'W'.

Usage:
    from config.variables_deposit_study import FRED_VARIABLES as FRED_VARIABLES_DEPOSITS
    # then pass FRED_VARIABLES_DEPOSITS to download_var_data(..., fred_variables=...)
"""

# --- Paper‑aligned macro block (cloned from variables_alternative.py) ---
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

    # === Interest Rates (levels; FFR omitted in paper baseline) ===
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
    },

    # === Deposits block (monthly; log levels ×100 unless noted) ===
    'DPSACBM027SBOG': {
        'name': 'Total Deposits, All Commercial Banks',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, Bil.$',
        'frequency': 'M'
    },
    'TCDSL': {
        'name': 'Total Checkable Deposits',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, Bil.$',
        'frequency': 'M'
    },
    'OCDSL': {
        'name': 'Other Checkable Deposits',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, Bil.$',
        'frequency': 'M'
    },
    'DEMDEPSL': {
        'name': 'Demand Deposits',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, Bil.$',
        'frequency': 'M'
    },
    'ODSACBM027SBOG': {
        'name': 'Other Deposits, All Commercial Banks',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, Bil.$',
        'frequency': 'M'
    },
    'STDCBSL': {
        'name': 'Small‑Denomination Time Deposits (Commercial Banks)',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, Bil.$',
        'frequency': 'M'
    },
    'DPSSCBM027SBOG': {
        'name': 'Deposits, Small Domestically Chartered Commercial Banks',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, Bil.$',
        'frequency': 'M'
    },
    'DPSLCBM027SBOG': {
        'name': 'Deposits, Large Domestically Chartered Commercial Banks',
        'transform': 'log_level_100',
        'adjustment_status': 'sa',
        'units': 'SA, Bil.$',
        'frequency': 'M'
    },

    # === Financial conditions & uncertainty ===
    'KCFSI': {
        'name': 'Kansas City Financial Stress Index',
        'transform': 'level',
        'adjustment_status': 'raw',
        'units': 'Index',
        'frequency': 'M'
    },
    'ANFCI': {
        'name': 'Adjusted National Financial Conditions Index (weekly)',
        'transform': 'level',
        'adjustment_status': 'raw',
        'units': 'Index',
        'frequency': 'D'  # weekly series; mark as high‑freq so it is averaged to Q
    },
    'USEPUINDXM': {
        'name': 'U.S. Economic Policy Uncertainty Index (monthly)',
        'transform': 'log_level_100',
        'adjustment_status': 'raw',
        'units': 'Index',
        'frequency': 'M'
    }
}

# Development subset focused on deposits add‑ons
DEV_FRED_VARIABLES = {
    'DPSACBM027SBOG': FRED_VARIABLES['DPSACBM027SBOG'],
    'TCDSL': FRED_VARIABLES['TCDSL'],
    'STDCBSL': FRED_VARIABLES['STDCBSL'],
    'KCFSI': FRED_VARIABLES['KCFSI'],
    'USEPUINDXM': FRED_VARIABLES['USEPUINDXM'],
    # include one macro + one rate for quick sanity
    'GDPC1': FRED_VARIABLES['GDPC1'],
    'DGS10': FRED_VARIABLES['DGS10']
}

# Alternative series for missing data (limited fallbacks for new series)
ALTERNATIVE_SERIES = {
    'DTWEXBGS': 'DTWEXM',                 # Broad USD index fallback
    'PNRGINDEXM': 'PALLFNFINDEXQ',        # Commodity index fallback

    # Deposits fallbacks: prefer NSA if SA missing
    'DPSACBM027SBOG': 'DPSACBM027NBOG',
    'ODSACBM027SBOG': 'ODSACBM027NBOG',
    'DPSSCBM027SBOG': 'DPSSCBM027NBOG',
    'DPSLCBM027SBOG': 'DPSLCBM027NBOG',
    'STDCBSL': 'STDCBNS',
    'TCDSL': 'TCDNS',
    'OCDSL': 'OCDNS',

    # FCI/uncertainty fallbacks
    'ANFCI': 'NFCI',                      # Adjusted → unadjusted NFCI
    'USEPUINDXM': 'USEPUINDXD'            # monthly → daily (will be averaged)
}

# Sample window consistent with paper baseline
DATA_START = '1973-01-01'
DATA_END = '2019-03-31'

# Frequency lists for quarterly conversion logic
QUARTERLY_VARIABLES = [
    'GDPC1', 'PCECC96', 'PRFIC1', 'PNFIC1', 'EXPGSC1', 'IMPGSC1',
    'GCEC1', 'FGCEC1', 'GDPDEF', 'PCEPI', 'PCEPILFE', 'COMPRNFB'
]

MONTHLY_VARIABLES = [
    'CPIAUCSL', 'CPILFESL', 'PAYEMS', 'UNRATE', 'INDPRO', 'TCU',
    'HOUST', 'DSPIC96', 'UMCSENT', 'AAA', 'BAA', 'PNRGINDEXM',
    # Deposits & FCI/EPU additions
    'DPSACBM027SBOG', 'TCDSL', 'OCDSL', 'DEMDEPSL', 'ODSACBM027SBOG',
    'STDCBSL', 'DPSSCBM027SBOG', 'DPSLCBM027SBOG',
    'KCFSI', 'USEPUINDXM'
]

DAILY_VARIABLES = [
    'DGS2', 'DGS5', 'DGS10', 'DTWEXBGS', 'SP500', 'DCOILWTICO', 'VIXCLS',
    # Treat weekly ANFCI as high‑frequency for quarterly averaging
    'ANFCI'
]

# Document notable exclusions (paper baseline omitted DFF)
EXCLUDED_SERIES = {
    'DFF': 'Federal funds effective rate omitted in paper baseline (ZLB considerations).'
}

