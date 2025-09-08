"""
Cholesky Ordering Configuration for VAR Models
Based on standard macroeconomic practice and economic theory

The ordering reflects the assumed speed of adjustment and contemporaneous causal structure:
- Most exogenous (slow-moving) variables first
- Most endogenous (fast-moving) variables last

This ensures proper identification of structural shocks in VAR models.
"""

# Standard macroeconomic ordering (most exogenous to most endogenous)
CHOLESKY_ORDERING = [
    # ===== REAL ACTIVITY BLOCK (Slow-moving) =====
    # Real GDP and components - fundamental economic activity
    'GDPC1',        # Real Gross Domestic Product
    'PCECC96',      # Real Personal Consumption Expenditures
    'EXPGSC1',      # Real Exports of Goods and Services
    'IMPGSC1',      # Real Imports of Goods and Services
    'GCEC1',        # Real Government Consumption and Investment
    'FGCEC1',       # Real Federal Government Consumption and Investment
    'PRFIC1',       # Real Private Residential Fixed Investment
    'PNFIC1',       # Real Private Nonresidential Fixed Investment
    
    # Industrial production and capacity
    'INDPRO',       # Industrial Production Index
    'TCU',          # Capacity Utilization: Manufacturing
    'HOUST',        # Housing Starts
    
    # ===== LABOR MARKET BLOCK (Medium-slow) =====
    'PAYEMS',       # All Employees, Total Nonfarm
    'UNRATE',       # Unemployment Rate
    'COMPRNFB',     # Nonfarm Business Sector: Compensation Per Hour
    
    # ===== INCOME AND CONSUMPTION =====
    'DSPIC96',      # Real Disposable Personal Income
    'UMCSENT',      # University of Michigan Consumer Sentiment
    
    # ===== PRICE LEVEL BLOCK (Medium speed) =====
    # GDP deflator and core price measures
    'GDPDEF',       # GDP Deflator
    'PCEPILFE',     # Core PCE Price Index (Fed's preferred measure)
    'PCEPI',        # PCE Price Index
    'CPILFESL',     # Core CPI
    'CPIAUCSL',     # CPI All Items
    
    # ===== FINANCIAL BLOCK (Fast-moving) =====
    # Short-term rates (monetary policy)
    'DFF',          # Federal Funds Rate
    'DGS2',         # 2-Year Treasury Rate
    
    # Long-term rates
    'DGS5',         # 5-Year Treasury Rate
    'DGS10',        # 10-Year Treasury Rate
    
    # Corporate rates
    'AAA',          # Moody's Seasoned Aaa Corporate Bond Yield
    'BAA',          # Moody's Seasoned Baa Corporate Bond Yield
    
    # ===== ASSET PRICES AND EXTERNAL (Fastest) =====
    'DTWEXBGS',     # Nominal Broad U.S. Dollar Index
    'SP500',        # S&P 500 Stock Index
    'DCOILWTICO',   # Crude Oil Prices
    'PNRGINDEXM',   # Global Non-Energy Commodity Price Index
    'VIXCLS',       # VIX Volatility Index
]

# Alternative ordering for monetary policy analysis
MONETARY_POLICY_ORDERING = [
    # Real activity block
    'GDPC1', 'INDPRO', 'PAYEMS', 'UNRATE',
    
    # Price block
    'CPIAUCSL', 'PCEPILFE',
    
    # Monetary policy block
    'DFF', 'DGS10',
    
    # Asset prices
    'SP500'
]

# Compact ordering for testing with key variables
COMPACT_ORDERING = [
    'GDPC1',        # Real GDP
    'CPIAUCSL',     # Inflation
    'UNRATE',       # Unemployment
    'DFF',          # Fed Funds Rate
    'DGS10',        # Long-term rate
    'SP500'         # Stock market
]

def get_ordering_for_variables(variable_list, ordering_type='standard'):
    """
    Get appropriate ordering for a given list of variables
    
    Parameters:
    -----------
    variable_list : list
        List of variable names in the VAR
    ordering_type : str
        Type of ordering ('standard', 'monetary', 'compact')
        
    Returns:
    --------
    list
        Ordered list of variables present in variable_list
    """
    if ordering_type == 'standard':
        reference_ordering = CHOLESKY_ORDERING
    elif ordering_type == 'monetary':
        reference_ordering = MONETARY_POLICY_ORDERING
    elif ordering_type == 'compact':
        reference_ordering = COMPACT_ORDERING
    else:
        raise ValueError(f"Unknown ordering type: {ordering_type}")
    
    # Filter to only include variables present in the data
    ordered_variables = [var for var in reference_ordering if var in variable_list]
    
    # Add any remaining variables at the end (maintaining their original order)
    remaining_variables = [var for var in variable_list if var not in ordered_variables]
    
    return ordered_variables + remaining_variables

def validate_ordering(variable_list, ordering):
    """
    Validate that ordering contains all variables from variable_list
    
    Parameters:
    -----------
    variable_list : list
        Complete list of variables in the VAR
    ordering : list
        Proposed ordering
        
    Returns:
    --------
    bool
        True if ordering is valid
        
    Raises:
    -------
    ValueError
        If ordering is invalid
    """
    if set(variable_list) != set(ordering):
        missing_in_ordering = set(variable_list) - set(ordering)
        extra_in_ordering = set(ordering) - set(variable_list)
        
        error_msg = "Invalid ordering:\n"
        if missing_in_ordering:
            error_msg += f"Missing variables: {missing_in_ordering}\n"
        if extra_in_ordering:
            error_msg += f"Extra variables: {extra_in_ordering}\n"
            
        raise ValueError(error_msg)
    
    return True

# Economic rationale for the ordering:
ORDERING_RATIONALE = {
    'real_activity': 'Real economic activity variables are placed first as they represent '
                    'fundamental economic conditions that adjust slowly to shocks.',
    
    'labor_market': 'Labor market variables come after real activity as employment '
                   'responds to output changes but with some delay.',
                   
    'prices': 'Price variables are in the middle as they respond to real activity '
             'but also influence monetary policy decisions.',
             
    'financial': 'Financial variables are placed last as they adjust quickly to '
                'news and policy changes, reflecting forward-looking behavior.',
                
    'policy_implication': 'This ordering allows monetary policy (DFF) to have '
                         'contemporaneous effects on asset prices and long-term rates, '
                         'but real activity and prices respond only with a lag.'
}