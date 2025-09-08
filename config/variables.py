"""
Variable configuration for the Large Bayesian VAR model
Based on Crump et al. (2024) - Table 1
"""

# FRED series IDs and transformations for the 31 variables
FRED_VARIABLES = {
    # Real GDP and Components (log differences*100 for stationarity)
    'GDPC1': {
        'name': 'Real Gross Domestic Product',
        'transform': 'log_diff_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },
    'PCECC96': {
        'name': 'Real Personal Consumption Expenditures',
        'transform': 'log_diff_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },
    'PRFIC1': {
        'name': 'Real Private Residential Fixed Investment',
        'transform': 'log_diff_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },
    'PNFIC1': {
        'name': 'Real Private Nonresidential Fixed Investment',
        'transform': 'log_diff_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },
    'EXPGSC1': {
        'name': 'Real Exports of Goods and Services',
        'transform': 'log_diff_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },
    'IMPGSC1': {
        'name': 'Real Imports of Goods and Services',
        'transform': 'log_diff_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },
    'GCEC1': {
        'name': 'Real Government Consumption Expenditures and Gross Investment',
        'transform': 'log_diff_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },
    'FGCEC1': {
        'name': 'Real Federal Government Consumption Expenditures and Gross Investment',
        'transform': 'log_diff_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'Q'
    },
    
    # Price Indices (log differences*100 for inflation rates)
    'GDPDEF': {
        'name': 'Gross Domestic Product: Chain-type Price Index',
        'transform': 'log_diff_100',
        'adjustment_status': 'sa',
        'units': 'SA, 2012=100',
        'frequency': 'Q'
    },
    'PCEPI': {
        'name': 'Personal Consumption Expenditures: Chain-type Price Index',
        'transform': 'log_diff_100',
        'adjustment_status': 'sa',
        'units': 'SA, 2012=100',
        'frequency': 'Q'
    },
    'PCEPILFE': {
        'name': 'Personal Consumption Expenditures Excluding Food and Energy (Core PCE)',
        'transform': 'log_diff_100',
        'adjustment_status': 'sa',
        'units': 'SA, 2012=100',
        'frequency': 'Q'
    },
    'CPIAUCSL': {
        'name': 'Consumer Price Index for All Urban Consumers: All Items',
        'transform': 'log_diff_100',
        'adjustment_status': 'sa',
        'units': 'SA, 1982-84=100',
        'frequency': 'M'
    },
    'CPILFESL': {
        'name': 'Consumer Price Index for All Urban Consumers: All Items Less Food and Energy',
        'transform': 'log_diff_100',
        'adjustment_status': 'sa',
        'units': 'SA, 1982-84=100',
        'frequency': 'M'
    },
    'COMPRNFB': {
        'name': 'Nonfarm Business Sector: Compensation Per Hour',
        'transform': 'log_diff_100',
        'adjustment_status': 'sa',
        'units': 'SA, Index 2012=100',
        'frequency': 'Q'
    },
    
    # Labor Market (various transformations)
    'PAYEMS': {
        'name': 'All Employees, Total Nonfarm',
        'transform': 'log_diff_100',
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
        'transform': 'log_diff_100',
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
        'transform': 'log_diff_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Thous.Units',
        'frequency': 'M'
    },
    'DSPIC96': {
        'name': 'Real Disposable Personal Income',
        'transform': 'log_diff_100',
        'adjustment_status': 'saar',
        'units': 'SAAR, Bil.Chn.2012$',
        'frequency': 'M'
    },
    'UMCSENT': {
        'name': 'University of Michigan: Consumer Sentiment',
        'transform': 'level',
        'adjustment_status': 'nsa',
        'units': 'NSA, Q1-66=100',
        'frequency': 'M'
    },
    
    # Interest Rates (raw, percent per annum)
    'DFF': {
        'name': 'Federal Funds Effective Rate',
        'transform': 'level',
        'adjustment_status': 'raw',
        'units': '% p.a.',
        'frequency': 'D'
    },
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
    
    # Financial and Commodity Variables (log differences*100 for stationarity)
    'DTWEXBGS': {
        'name': 'Nominal Broad U.S. Dollar Index',
        'transform': 'log_diff_100',
        'adjustment_status': 'raw',
        'units': 'Mar-73=100',
        'frequency': 'D'
    },
    'SP500': {
        'name': 'S&P 500',
        'transform': 'log_diff_100',
        'adjustment_status': 'raw',
        'units': '1941-43=10',
        'frequency': 'D'
    },
    'DCOILWTICO': {
        'name': 'Crude Oil Prices: West Texas Intermediate',
        'transform': 'log_diff_100',
        'adjustment_status': 'raw',
        'units': '$/Barrel',
        'frequency': 'D'
    },
    'PNRGINDEXM': {
        'name': 'Global Price Index of All Commodities, Non-Energy',
        'transform': 'log_diff_100',
        'adjustment_status': 'raw',
        'units': 'AVG, Jan-02-70=100',
        'frequency': 'M'
    },
    
    # Volatility (raw)
    'VIXCLS': {
        'name': 'CBOE Volatility Index: VIX',
        'transform': 'level',
        'adjustment_status': 'raw',
        'units': 'Index',
        'frequency': 'D'
    }
}

DEV_FRED_VARIABLES = {
    'GDPC1': FRED_VARIABLES['GDPC1'],
    'CPIAUCSL': FRED_VARIABLES['CPIAUCSL'],
    'DFF': FRED_VARIABLES['DFF']
}


# Alternative series for missing data
ALTERNATIVE_SERIES = {
    'DTWEXBGS': 'DTWEXM',  # If broad index not available, use major currencies
    'PNRGINDEXM': 'PALLFNFINDEXQ',  # Alternative commodity index
}

# Data start and end dates (paper baseline uses 1973:Q1–2019:Q1)
DATA_START = '1973-01-01'
DATA_END = '2019-03-31'

# Frequency conversion settings
QUARTERLY_VARIABLES = ['GDPC1', 'PCECC96', 'PRFIC1', 'PNFIC1', 'EXPGSC1', 'IMPGSC1', 
                      'GCEC1', 'FGCEC1', 'GDPDEF', 'PCEPI', 'PCEPILFE', 'COMPRNFB']

MONTHLY_VARIABLES = ['CPIAUCSL', 'CPILFESL', 'PAYEMS', 'UNRATE', 'INDPRO', 'TCU', 
                    'HOUST', 'DSPIC96', 'UMCSENT', 'AAA', 'BAA', 'PNRGINDEXM']

DAILY_VARIABLES = ['DFF','DGS2', 'DGS5', 'DGS10', 'DTWEXBGS', 'SP500', 'DCOILWTICO', 'VIXCLS']
