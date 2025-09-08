#!/usr/bin/env python3
"""
Example: Proper IRF Analysis with Economic Ordering

This script demonstrates the correct way to perform impulse response 
analysis with proper Cholesky ordering and data transformations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from data_utils import load_var_data
from var_model import BayesianVAR
from irf_analysis import analyze_irfs
from config.cholesky_ordering import get_standard_ordering

def main():
    """
    Demonstrate proper IRF analysis workflow.
    """
    print("=== Proper IRF Analysis Example ===\n")
    
    # 1. Load data
    print("1. Loading data...")
    data = load_var_data()
    print(f"   Loaded {data.shape[0]} observations, {data.shape[1]} variables")
    
    # 2. Select key macroeconomic variables for demonstration
    key_variables = [
        # Real activity
        'GDPC1',      # Real GDP
        'INDPRO',     # Industrial production
        
        # Labor
        'PAYEMS',     # Nonfarm payrolls
        'UNRATE',     # Unemployment rate
        
        # Prices  
        'CPIAUCSL',   # Consumer price index
        'GDPDEF',     # GDP deflator
        
        # Financial
        'DFF',        # Federal funds rate
        'DGS10',      # 10-year Treasury
    ]
    
    # 3. Prepare analysis dataset
    print("\n2. Preparing analysis dataset...")
    available_vars = [var for var in key_variables if var in data.columns]
    model_data = data[available_vars].dropna()
    
    print(f"   Using variables: {available_vars}")
    print(f"   Sample size: {len(model_data)} observations")
    print(f"   Date range: {model_data.index[0]} to {model_data.index[-1]}")
    
    # 4. Verify ordering
    print("\n3. Applying economic ordering...")
    ordering = get_standard_ordering(available_vars)
    print(f"   Economic order: {ordering}")
    
    # 5. Estimate VAR model
    print("\n4. Estimating VAR model...")
    var_model = BayesianVAR(model_data, max_lags=4)  # Smaller model for example
    results = var_model.estimate()
    
    print(f"   Optimal lags: {results.k_ar}")
    print(f"   Log-likelihood: {results.llf:.2f}")
    
    # 6. Compute IRFs with proper ordering
    print("\n5. Computing impulse response functions...")
    print("   Using standard economic ordering (default)")
    
    # This uses economic ordering by default
    irf_results = analyze_irfs(
        results,
        periods=12,
        identification='cholesky',
        save_plots=True,
        output_dir='./examples/plots',
        use_standard_ordering=True  # This is the default
    )
    
    # 7. Validate key economic relationships
    print("\n6. Validating economic relationships...")
    
    # Check UNRATE response to GDPC1 shock
    if 'UNRATE' in available_vars and 'GDPC1' in available_vars:
        gdp_shock_responses = irf_results.irf[:, available_vars.index('UNRATE'), available_vars.index('GDPC1')]
        
        print(f"   UNRATE response to GDPC1 shock:")
        for period in [1, 3, 6, 12]:
            if period < len(gdp_shock_responses):
                response = gdp_shock_responses[period]
                direction = "↓" if response < 0 else "↑"
                print(f"     Period {period}: {response:.4f} {direction}")
        
        # Economic validation
        min_response = np.min(gdp_shock_responses[1:6])  # Periods 1-5
        if min_response < 0:
            print("   ✅ Economic validation: Unemployment decreases after GDP shock")
        else:
            print("   ⚠️  Economic concern: Unemployment should decrease after positive GDP shock")
    
    # 8. Generate summary
    print("\n7. Analysis Summary:")
    print(f"   - Model: VAR({results.k_ar}) with {len(available_vars)} variables")
    print(f"   - Identification: Cholesky with economic ordering")
    print(f"   - Sample: {len(model_data)} observations")
    print(f"   - Results saved to: ./examples/plots/")
    
    print("\n=== Analysis Complete ===")
    print("\nNext steps:")
    print("1. Review plots in ./examples/plots/")
    print("2. Check IRF CSV files for detailed results")
    print("3. Validate all economic relationships")
    print("4. Consider data transformations if needed")

if __name__ == "__main__":
    main()