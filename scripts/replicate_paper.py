import numpy as np
import pandas as pd
import sys
import os

# Add src to path to allow for module import
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bayesian.core_bayesian_var import LargeBayesianVAR

def main():
    """
    Script to replicate key results from Crump et al. (2021) paper.
    """
    # --- 1. Define Variables and Transformations ---
    # Based on Table 1 and Section 2.1 of the paper
    # Variables that enter in levels (raw)
    levels_vars = [
        'UNRATE', 'DGS2', 'DGS5', 'DGS10', 'AAA', 'BAA', 'DTWEXBGS', 'SP500',
        'VIXCLS', 'DCOILWTICO', 'PNRGINDEXM'
    ]
    # Variables that enter in 100 * log-levels
    log_levels_vars = [
        'GDPC1', 'PCECC96', 'PRFIC1', 'PNFIC1', 'EXPGSC1', 'IMPGSC1', 'GCEC1',
        'FGCEC1', 'GDPDEF', 'PCEPI', 'PCEPILFE', 'CPIAUCSL', 'CPILFESL',
        'COMPRNFB', 'PAYEMS', 'INDPRO', 'TCU', 'HOUST', 'DSPIC96', 'UMCSENT'
    ]
    
    all_vars = levels_vars + log_levels_vars
    
    # --- 2. Load and Transform Data ---
    print("--- Loading and transforming data ---")
    try:
        df = pd.read_csv('data/var_data_crump2024_crump2024.csv', index_col=0, parse_dates=True)
        
        # Select only the required variables
        data = df[all_vars]
        
        # The data in this CSV is already transformed according to Crump et al. (2024) methodology.
        # No further transformations are needed here.
            
        # Drop rows with any NaN values to ensure a clean dataset for the VAR.
        # This will align the start date of the data for all series.
        data = data.dropna()
        
        print("Data loaded and transformed successfully. Shape:", data.shape)
        print("Data start date:", data.index.min())
        print("Data end date:", data.index.max())
        print(data.head())
        
    except FileNotFoundError:
        print("Error: Data file 'data/var_data_crump2024_crump2024.csv' not found.")
        print("Please ensure the data file is in the correct location.")
        return # Exit if data not found

    # --- 3. Instantiate and Fit the BVAR Model ---
    print("\n--- Initializing and fitting the BVAR model (this may take a while) ---")
    # Paper states n_lags = 5
    bvar_model = LargeBayesianVAR(n_lags=5)
    
    # Use more draws for replication, but keep reasonable for a test run
    # Paper uses 50000 draws, 25000 burn-in. Let's use 5000/2000 for initial test.
    bvar_model.fit(data.values, n_draws=1000, n_burnin=500)
    
    print("\n--- Model fitting complete ---")
    print(f"Shape of beta draws: {bvar_model.beta_draws.shape}")
    print(f"Shape of sigma draws: {bvar_model.sigma_draws.shape}")

    # --- 4. Replicate Figure 6: Responses to a 1% increase in Real GDP ---
    print("\n--- Replicating Figure 6: Responses to a 1% increase in Real GDP ---")
    # Figure 6 is a generalized impulse response function (GIRF) to GDP.
    # In the paper, it's implemented as a conditional forecast:
    # "increase the one-period ahead forecasts for real GDP by one percent,
    # while leaving all other variables unconstrained."
    
    # Find the index of GDPC1
    gdp_idx = data.columns.get_loc('GDPC1')
    
    # Define the constraint: GDPC1 (index gdp_idx) is 1.0 (1 percent) at horizon 1 (t+1)
    # The conditional forecast function expects constraints as {horizon_step: {var_idx: value}}
    # Note: The paper says "one percent increase in the level". Since our data is 100*log-levels,
    # a 1 unit increase in 100*log(GDP) is approximately a 1% increase in GDP.
    constraints = {
        1: {gdp_idx: data['GDPC1'].iloc[-1] + 1.0} # Constrain GDPC1 to be 1 unit higher than its last observed value
    }
    
    # Forecast horizon for IRFs in the paper is typically 20-40 quarters. Let's use 40.
    horizon = 40
    
    # Get conditional forecast draws
    cf_draws = bvar_model.get_conditional_forecast(horizon, constraints)
    
    # To get the "response" as in Figure 6, we need to subtract the unconditional forecast.
    # First, get unconditional forecasts.
    # The conditional_forecast function can also generate unconditional forecasts if constraints are empty.
    unconditional_forecast_draws = bvar_model.get_conditional_forecast(horizon, {}) 
    
    # Calculate the difference (response) for each draw
    response_draws = cf_draws - unconditional_forecast_draws
    
    # Summarize the responses (median and credible intervals)
    # Reshape for summary functions: (n_draws, horizon, n_vars) -> (n_draws, horizon * n_vars)
    n_draws, horizon, n_vars = response_draws.shape
    response_reshaped = response_draws.reshape(n_draws, -1)
    
    response_summary = bvar_model.credible_intervals.get_posterior_summary(response_reshaped)
    response_intervals = bvar_model.credible_intervals.get_credible_intervals(response_reshaped)
    
    # Reshape back to (n_vars, horizon) for easier access (response_var, horizon)
    median_response = response_summary['median'].reshape(horizon, n_vars).T # Transpose to (n_vars, horizon)
    
    print(f"\nMedian Response of GDPC1 to a 1% increase in GDPC1 (first 10 periods):")
    print(np.round(median_response[gdp_idx, :10], 4))
    
    print(f"\nMedian Response of PCEPI to a 1% increase in GDPC1 (first 10 periods):")
    pcepi_idx = data.columns.get_loc('PCEPI')
    print(np.round(median_response[pcepi_idx, :10], 4))
    
    # You would typically plot these to compare with Figure 6.
    # For now, we just print a few values.

    print("\n--- Replication attempt complete ---")

if __name__ == '__main__':
    main()
