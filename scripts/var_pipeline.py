import os
import json
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.stattools import jarque_bera
import sys

# Add config directory to path for importing plot utilities
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
from plot_utils import format_variable_name, format_variable_names
# We avoid requiring the python‑dotenv package by implementing a simple
# .env parser.  Each line in the .env file should be of the form KEY=VALUE.

"""
This script implements a pipeline for downloading macroeconomic data from the
Federal Reserve Economic Database (FRED), estimating a vector autoregression
(VAR) model, and producing impulse response functions (IRFs), forecast-error
variance decompositions (FEVD), and standard diagnostics.  The design of the
pipeline roughly follows the data and transformation conventions described in
Crump et al. (2025)【897288989664335†L298-L479】—rate series enter in levels while
all other series are transformed to scaled log levels (100×log).

To run the script you must supply a valid FRED API key via a .env file in the
same directory as this script with an entry of the form:

    FRED_API_KEY=your_api_key_here

See https://fred.stlouisfed.org/docs/api/api_key.html for instructions on
obtaining an API key.  Once the .env file is configured, execute this script
with:

    python var_pipeline.py

The script will download each series, align them to a common quarterly
frequency (averaging higher‑frequency observations when necessary), transform
the series appropriately, fit a VAR model (with lag order selected by AIC),
and produce plots of IRFs, FEVDs, and diagnostic statistics.  All plots
are saved into a `figures/` subdirectory.
"""

# Define the series to download.  The keys are FRED series identifiers, the
# values are tuples containing a human‑readable variable name and a flag
# indicating whether the series should be logged ('log') or left in levels
# ('level').  This selection loosely follows the variables documented in
# Table 1 of Crump et al. (2025)【897288989664335†L298-L479】, but it excludes a
# handful of specialized financial or commodity series for simplicity.
SERIES_INFO = {
    "GDPC1": ("Real GDP", "log"),
    "PCECC96": ("Real Consumption", "log"),
    "PRFIC1": ("Real Residential Investment", "log"),
    "PNFIC1": ("Real Nonresidential Investment", "log"),
    "EXPGSC1": ("Real Exports", "log"),
    "IMPGSC1": ("Real Imports", "log"),
    "GCEC1": ("Real Government Spending", "log"),
    # Federal spending omitted because it is highly collinear with GCEC1
    "GDPDEF": ("GDP Deflator", "log"),
    "PCEPI": ("PCE Price Index", "log"),
    "CPIAUCSL": ("CPI All Items", "log"),
    "CPILFESL": ("CPI Less Food & Energy", "log"),
    "COMPNFB": ("Compensation per Hour", "log"),
    "PAYEMS": ("Total Nonfarm Payrolls", "log"),
    "UNRATE": ("Unemployment Rate", "level"),
    "INDPRO": ("Industrial Production", "log"),
    "MCUMFN": ("Manufacturing Capacity Utilization", "level"),
    "HOUST": ("Housing Starts", "log"),
    "DSPIC96": ("Real Disposable Personal Income", "log"),
    "UMCSENT": ("Consumer Sentiment", "level"),
    "GS2": ("2‑Year Treasury Yield", "level"),
    "GS5": ("5‑Year Treasury Yield", "level"),
    "GS10": ("10‑Year Treasury Yield", "level"),
    "AAA": ("Moody's Aaa Bond Yield", "level"),
    "BAA": ("Moody's Baa Bond Yield", "level"),
    "DTWEXM": ("Trade‑Weighted USD", "log"),
    "SP500": ("S&P 500 Index", "log"),
    "DCOILWTICO": ("WTI Oil Price", "log"),
    # Substitute for the S&P GSCI non‑energy index: global non‑fuel commodity price index
    "PNFUELINDEXQ": ("Global Non‑Fuel Commodity Price Index", "log"),
}
# Configuration constants
MIN_OBS_FOR_VAR = 80          # target minimum T after pruning
MAX_USER_LAGS = 8             # user cap for lag order search
DETERMINISTICS = 1            # intercept only
REPORT_DIR = "figures"        # reuse figures dir for CSV reports

def load_api_key() -> str:
    """Load the FRED API key from a .env file.

    This function looks for a file named `.env` in the current working
    directory.  Each line of the file should have the form KEY=VALUE.
    Lines beginning with `#` are ignored.  The variable FRED_API_KEY is
    extracted and returned.  If the file or key is missing, a
    RuntimeError is raised.
    """
    # Attempt to read the .env file from the directory where the script resides
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    api_key = None
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "FRED_API_KEY":
                        api_key = value
                        break
    # Fallback to environment variable if not found in the file
    if not api_key:
        api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED API key not found.  Please create a .env file with FRED_API_KEY or set it in the environment."
        )
    return api_key

def fetch_fred_series(series_id: str, api_key: str,
                      frequency: str = "q",
                      aggregation_method: str = "avg",
                      start_date: str = "1950-01-01") -> pd.Series:
    """
    Fetch a series from FRED and return a pandas Series indexed by date.

    Parameters
    ----------
    series_id : str
        FRED series identifier.
    api_key : str
        User's FRED API key.
    frequency : {'d','w','m','q','sa','a'}, default 'q'
        Desired frequency of returned data.  'q' denotes quarterly.
    aggregation_method : {'avg','sum','eop'}, default 'avg'
        Method for aggregating higher frequency observations to the desired
        frequency.  'avg' takes the average, 'sum' sums the values and
        'eop' takes the last observation of the period.
    start_date : str, default '1950-01-01'
        Beginning of the observation window (YYYY-MM-DD).

    Returns
    -------
    pandas.Series
        A time series of observations converted to numeric values and indexed
        by pandas.Timestamp objects.  Missing values are represented as NaN.
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "frequency": frequency,
        "aggregation_method": aggregation_method,
        "observation_start": start_date
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch {series_id}: status {response.status_code}")
    try:
        data = response.json()["observations"]
    except (json.JSONDecodeError, KeyError) as exc:
        # If the API returned an error message instead of JSON, propagate it
        raise RuntimeError(f"Error parsing JSON response for {series_id}") from exc
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df["value"]

def build_dataset(series_info: dict, api_key: str) -> pd.DataFrame:
    """
    Download all series defined in SERIES_INFO, apply the prescribed
    transformation (log or level), and align them on a common quarterly
    index.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the transformed data for all variables.
    """
    dfs = []
    for series_id, (name, xform) in series_info.items():
        print(f"Downloading {name} ({series_id})...")
        # For interest rates we want the end‑of‑period value rather than the average
        # so we use 'eop' for rate series and 'avg' for others.
        method = "eop" if xform == "level" else "avg"
        try:
            s = fetch_fred_series(series_id, api_key, frequency="q", aggregation_method=method)
        except RuntimeError as exc:
            # If a series fails, warn and continue
            print(str(exc))
            continue
        # Apply transformation: 100×log for log series, raw level otherwise.
        if xform == "log":
            s = np.log(s) * 100.0
        dfs.append(s.rename(name))
def compute_feasible_maxlags(T: int, k: int, d: int = DETERMINISTICS) -> int:
    """
    Conservative feasibility bound for VAR lag order given sample size T,
    number of variables k, and deterministics count d.

    Uses a simple rule of thumb: T > (k + 1) * p + d + 1  =>  p <= floor((T - d - 1) / (k + 1))
    """
    if k <= 0:
        return 0
    return max(0, int((T - d - 1) // (k + 1)))


def compute_coverage_report(df: pd.DataFrame, report_name: str = "coverage") -> tuple[pd.DataFrame, int, int]:
    """
    Compute coverage stats for each series:
      - first_valid: first non-missing timestamp
      - last_valid: last non-missing timestamp
      - n_obs: number of non-missing observations

    Also prints and returns (report_df, T_intersection, k_intersection).
    Saves CSV to REPORT_DIR/{report_name}.csv
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    rows = []
    for col in df.columns:
        s = df[col].dropna()
        first = s.index.min() if not s.empty else pd.NaT
        last = s.index.max() if not s.empty else pd.NaT
        n = int(s.shape[0])
        rows.append({"variable": col, "first_valid": first, "last_valid": last, "n_obs": n})

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(by=["first_valid", "n_obs"], ascending=[False, False])

    # Intersection sample length if we were to drop all rows with any NaN
    sample = df.dropna()
    T, k = sample.shape if not sample.empty else (0, df.shape[1])
    print(f"[Coverage] {report_name}: variables={df.shape[1]}, intersection T={T}, k={k}")

    csv_path = os.path.join(REPORT_DIR, f"{report_name}.csv")
    report.to_csv(csv_path, index=False)
    return report, T, k


def prune_to_min_obs(df: pd.DataFrame, min_obs: int = MIN_OBS_FOR_VAR) -> tuple[pd.DataFrame, list[str]]:
    """
    Iteratively drop the latest-starting series until the intersection sample
    length (after dropna) is at least min_obs. Returns (pruned_df, dropped_list).
    Also writes REPORT_DIR/prune_report.csv with kept/dropped lists.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    df_work = df.copy()
    dropped: list[str] = []

    while True:
        sample = df_work.dropna()
        T = int(sample.shape[0])
        if T >= min_obs:
            break

        # If we have no columns left or nothing to compute coverage on, stop
        if df_work.shape[1] == 0:
            print("[Prune] No variables remain to drop.")
            break

        # Identify the latest-starting variable(s)
        coverage_rows = []
        for col in df_work.columns:
            s = df_work[col].dropna()
            first = s.index.min() if not s.empty else pd.NaT
            n = int(s.shape[0])
            coverage_rows.append({"variable": col, "first_valid": first, "n_obs": n})

        cov = pd.DataFrame(coverage_rows)
        # Sort by latest start (descending first_valid), and for ties keep the shortest series (ascending n_obs)
        cov = cov.sort_values(by=["first_valid", "n_obs"], ascending=[False, True]).reset_index(drop=True)

        to_drop = cov.loc[0, "variable"]
        firstv = cov.loc[0, "first_valid"]
        nobs = int(cov.loc[0, "n_obs"])
        firstv_str = firstv.date().isoformat() if pd.notna(firstv) else "NaT"
        print(f"[Prune] Dropping '{to_drop}' starting at {firstv_str} (n={nobs}); intersection T={T} &lt; {min_obs}")

        df_work = df_work.drop(columns=[to_drop])
        dropped.append(to_drop)

        # Safety: break if only one series remains and still not enough T
        if df_work.shape[1] <= 1:
            break

    kept = [c for c in df.columns if c not in dropped]
    # Save a simple report aligning kept/dropped into two columns
    max_len = max(len(kept), len(dropped))
    kept_ext = kept + [""] * (max_len - len(kept))
    dropped_ext = dropped + [""] * (max_len - len(dropped))
    prune_report = pd.DataFrame({"kept": kept_ext, "dropped": dropped_ext})
    prune_report.to_csv(os.path.join(REPORT_DIR, "prune_report.csv"), index=False)

    return df_work, dropped
    # Combine into a single DataFrame using an outer join on the index
    data = pd.concat(dfs, axis=1)
    # Drop rows with any missing observations
    data = data.dropna()
    return data

def estimate_var(data: pd.DataFrame, maxlags: int = 8) -> VAR:
    """
    Fit a VAR model to the provided data.

    Parameters
    ----------
    data : pandas.DataFrame
        Time series data (T×k) after transformation.
    maxlags : int, default 8
        Maximum number of lags to consider when selecting the order via AIC.

    Returns
    -------
    VARResults
        Estimated VAR model.
    """
    model = VAR(data)
    # Determine lag order by AIC
    order_results = model.select_order(maxlags=maxlags)
    selected_lag = order_results.aic
    print(f"Selected lag order by AIC: {selected_lag}")
    var_results = model.fit(selected_lag)
    return var_results

def plot_irfs(var_results, horizon: int = 20, outdir: str = "figures"):
    """
    Plot impulse response functions for each variable in the VAR.

    Parameters
    ----------
    var_results : VARResults
        Fitted VAR model.
    horizon : int, default 20
        Number of steps ahead to compute impulse responses.
    outdir : str, default 'figures'
        Directory to save the generated plots.
    """
    os.makedirs(outdir, exist_ok=True)
    irf = var_results.irf(horizon)
    # The irf.plot function returns a matplotlib.Figure.  Use it to
    # create a combined grid of plots.  Set cumulative=False to
    # display responses in levels.
    fig = irf.plot(orth=True, impulse=None, response=None)
    fig.suptitle("Impulse Response Functions")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "irf_all.png"), dpi=300)
    plt.close(fig)

def plot_fevd(var_results, horizon: int = 20, outdir: str = "figures"):
    """
    Plot forecast error variance decomposition (FEVD) for each variable.

    Parameters
    ----------
    var_results : VARResults
        Fitted VAR model.
    horizon : int, default 20
        Number of periods ahead for the FEVD.
    outdir : str, default 'figures'
        Directory to save the generated plots.
    """
    os.makedirs(outdir, exist_ok=True)
    fevd = var_results.fevd(horizon)
    # Create a stacked bar chart for each variable at a given horizon (e.g., 10)
    # We'll use the last horizon by default.
    target_horizon = horizon - 1
    fevd_data = fevd.decomp[:, target_horizon, :]
    labels = var_results.names
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(labels))
    for i, shock in enumerate(labels):
        ax.bar(labels, fevd_data[:, i], bottom=bottom, label=shock)
        bottom += fevd_data[:, i]
    ax.set_ylabel("Proportion")
    ax.set_title(f"Forecast Error Variance Decomposition (h = {horizon})")
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fevd_h{horizon}.png"), dpi=300)
    plt.close(fig)

def run_diagnostics(var_results, outdir: str = "figures"):
    """
    Compute and plot residual diagnostics, including autocorrelation,
    normality, and heteroskedasticity tests.

    Parameters
    ----------
    var_results : VARResults
        Fitted VAR model.
    outdir : str, default 'figures'
        Directory to save the generated diagnostic plots.
    """
    os.makedirs(outdir, exist_ok=True)
    resid = var_results.resid

    # 1. Autocorrelation: Ljung–Box statistic for each residual series
    for i, col in enumerate(resid.columns):
        lb = acorr_ljungbox(resid[col], lags=[12], return_df=True)
        print(f"Ljung–Box Q(12) for {col}: statistic={lb['lb_stat'].iloc[0]:.2f}, p={lb['lb_pvalue'].iloc[0]:.3f}")

    # 2. Heteroskedasticity: ARCH LM test
    for i, col in enumerate(resid.columns):
        lm_stat, lm_pvalue, _, _ = het_arch(resid[col], nlags=4)
        print(f"ARCH LM(4) for {col}: statistic={lm_stat:.2f}, p={lm_pvalue:.3f}")

    # 3. Normality: Jarque–Bera test
    for i, col in enumerate(resid.columns):
        jb_stat, jb_pvalue, _, _ = jarque_bera(resid[col])
        print(f"Jarque–Bera for {col}: statistic={jb_stat:.2f}, p={jb_pvalue:.3f}")

    # 4. Durbin–Watson statistic for serial correlation across equations
    dw_stats = durbin_watson(resid)
    for name, dw in zip(resid.columns, dw_stats):
        print(f"Durbin–Watson for {name}: {dw:.2f}")

    # Plot autocorrelation functions of residuals
    from statsmodels.graphics.tsaplots import plot_acf
    for col in resid.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_acf(resid[col], lags=20, ax=ax, alpha=0.05)
        ax.set_title(f"ACF of residuals: {format_variable_name(col)}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"acf_resid_{col}.png"), dpi=300)
        plt.close(fig)

def main():
    api_key = load_api_key()
    data = build_dataset(SERIES_INFO, api_key)
    # Basic descriptive statistics
    print(data.describe().T)
    # Estimate the VAR
    var_results = estimate_var(data)
    # Generate impulse responses and FEVD
    plot_irfs(var_results, horizon=20, outdir="figures")
    plot_fevd(var_results, horizon=20, outdir="figures")
    # Run diagnostics
    run_diagnostics(var_results, outdir="figures")
    print("Pipeline completed.  Figures saved in the 'figures' directory.")

if __name__ == "__main__":
    main()