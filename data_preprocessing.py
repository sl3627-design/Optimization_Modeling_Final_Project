import pandas as pd
import numpy as np

# Load raw datasets (S&P 500 constituents - Current)
stock_df = pd.read_csv("stock_daily_2017_2025.csv")
fund_df = pd.read_csv("fundamentals_q_2017_2025.csv")

# =========================================================
# 1. Pre-process Daily Stock Data
# =========================================================
# Drop rows with missing stock prices or returns
stock_df.dropna(subset=['DlyPrc', 'DlyRet', 'DlyRetx'], inplace=True)

# Convert string dates to datetime objects (Format: YYYY-MM-DD)
stock_df['DlyCalDt'] = pd.to_datetime(stock_df['DlyCalDt'], format='%Y-%m-%d')

# Rename columns to standardize the merge keys across datasets
stock_df.rename(columns={'Ticker': 'tic', 'DlyCalDt': 'date'}, inplace=True)

# Sort strictly chronologically by Ticker, then Date
stock_df.sort_values(by=['tic', 'date'], inplace=True)

# Output cleaned stock data
stock_df.to_csv("cleaned_stock_daily.csv", index=False)


# =========================================================
# 2. Pre-process Quarterly Fundamentals Data
# =========================================================
# Drop rows missing the fundamental reporting date (rdq), total assets (atq), and net income (niq).
fund_df.dropna(subset=['rdq', 'atq', 'niq'], inplace=True)

# Convert strings to datetime objects (Format: DD/MM/YYYY)
fund_df['datadate'] = pd.to_datetime(fund_df['datadate'], format='%d/%m/%Y', errors='coerce')
fund_df['rdq'] = pd.to_datetime(fund_df['rdq'], format='%d/%m/%Y', errors='coerce')

# Drop any rows where dates failed to parse correctly
fund_df.dropna(subset=['datadate', 'rdq'], inplace=True)

# ---------------------------------------------------------
# Convert YTD cash flow items to single-quarter figures
# ---------------------------------------------------------
# Compustat's *cfy items (oancfy, fincfy, ivncfy) are YEAR-TO-DATE:
#   Q1 = Q1 flow, Q2 = Q1+Q2, Q3 = Q1+Q2+Q3, Q4 = full fiscal year.
# We derive true single-quarter flows so they align with niq (which is already quarterly).
# Sort by firm-fyear-fqtr so shift(1) is the prior quarter within the same fiscal year.
fund_df.sort_values(by=['tic', 'fyearq', 'fqtr'], inplace=True)

for ytd_col, q_col in [('oancfy', 'oancfq'), ('fincfy', 'fincfq'), ('ivncfy', 'ivncfq')]:
    prior_ytd = fund_df.groupby(['tic', 'fyearq'])[ytd_col].shift(1)
    # Q1: there is no prior quarter in the fiscal year, so oancfq = oancfy directly.
    # Q2-Q4: oancfq = oancfy - (prior YTD value).
    fund_df[q_col] = np.where(fund_df['fqtr'] == 1, fund_df[ytd_col], fund_df[ytd_col] - prior_ytd)

# Filter down to the essential quantitative features to reduce noise
cols_to_keep = ['GVKEY', 'tic', 'datadate', 'rdq', 'fyearq', 'fqtr', 'datacqtr', 
                'atq', 'ltq', 'niq', 'epspxq', 'fincfy', 'ivncfy', 'oancfy',
                'fincfq', 'ivncfq', 'oancfq']
fund_df = fund_df[cols_to_keep]

# Handle duplicate rows (Compustat occasionally outputs multiple formatting codes per quarter)
fund_df.sort_values(by=['tic', 'datadate', 'rdq'], inplace=True)
fund_df.drop_duplicates(subset=['tic', 'datadate'], keep='last', inplace=True)

# Output cleaned fundamentals data
fund_df.to_csv("cleaned_fundamentals_q.csv", index=False)


# ==============================================================
# 3. Merge Daily Stock Data and Quarterly Fundamentals Data
# ==============================================================
# Load the cleaned datasets
stock_df = pd.read_csv("cleaned_stock_daily.csv")
fund_df = pd.read_csv("cleaned_fundamentals_q.csv")

# Convert strings to datetime objects for accurate sorting
stock_df['date'] = pd.to_datetime(stock_df['date'])
fund_df['rdq'] = pd.to_datetime(fund_df['rdq'])
fund_df['datadate'] = pd.to_datetime(fund_df['datadate'])

# CRUCIAL: merge_asof requires both datasets to be sorted precisely by the time key
stock_df.sort_values(['date', 'tic'], inplace=True)
fund_df.sort_values(['rdq', 'tic'], inplace=True)

# Perform point-in-time merge (Look-Ahead Bias protection)
# For every row in stock_df, find the latest row in fund_df where fund_df.rdq <= stock_df.date
merged_df = pd.merge_asof(
    left=stock_df,
    right=fund_df,
    left_on='date',
    right_on='rdq',
    by='tic',
    direction='backward'
)

# Sort back to standard Ticker -> Date hierarchy for chronological modeling
merged_df.sort_values(['tic', 'date'], inplace=True)

# Drop early rows before a company published its first fundamental report in the dataset (early 2017)
merged_df.dropna(subset=['atq', 'niq'], inplace=True)

# Save the final merged dataset
merged_df.to_csv("merged_daily_fundamentals.csv", index=False)


# ==============================================================
# 4. Data Winsorization to Remove Severe Outliers
# ==============================================================
# Load the merged dataset containing daily returns and forward-filled fundamentals
df = pd.read_csv("merged_daily_fundamentals.csv")

# Calculate the 1st and 99th percentiles for the daily returns column (Removing outliers)
lower_bound = df['DlyRet'].quantile(0.01)
upper_bound = df['DlyRet'].quantile(0.99)

# Apply winsorization using the pandas .clip() function
df['DlyRet'] = df['DlyRet'].clip(lower=lower_bound, upper=upper_bound)

# Save the clean, winsorized dataset for model building
df.to_csv("merged_daily_fundamentals_winsorized.csv", index=False)



# ==============================================================
# 5. Mitigate Look-Ahead-Bias
# ==============================================================
# Load the winsorized, merged dataset
df = pd.read_csv("merged_daily_fundamentals_winsorized.csv")

# ==========================================
# 5-1. Ticker Anonymization
# ==========================================
# Extract all unique tickers and create an ID mapping
unique_tickers = df['tic'].unique()
ticker_mapping = {ticker: f"stock_{i+1}" for i, ticker in enumerate(unique_tickers)}

# Map the real tickers to the anonymized IDs
df['stock_id'] = df['tic'].map(ticker_mapping)

# Save the mapping key to a separate file (DO NOT give this to the LLM)
mapping_df = pd.DataFrame(list(ticker_mapping.items()), columns=['Original_Ticker', 'Stock_ID'])
mapping_df.to_csv("ticker_anonymization_key.csv", index=False)


# ==========================================
# 5-2. No Narrative Inputs
# ==========================================
# Drop all columns that could leak identity or narrative data.
# 'tic' is the ticker, 'GVKEY' is Compustat's internal ID, 'PERMNO' and 'HdrCUSIP' are CRSP IDs.
cols_to_drop = ['tic', 'GVKEY', 'PERMNO', 'PERMCO', 'HdrCUSIP']
cols_to_drop = [c for c in cols_to_drop if c in df.columns] # Only drop if they exist
df.drop(columns=cols_to_drop, inplace=True)

# Reorder columns to put the new 'stock_id' at the front for clean LLM prompting
cols = ['stock_id'] + [c for c in df.columns if c != 'stock_id']
df = df[cols]

# Save the strictly numerical, anonymized dataset
df.to_csv("llm_ready_dataset.csv", index=False)




# ==================================================================
# 6. Compute and Augment Input Metrics Used for Asset Selection
# ==================================================================
# Load the anonymized, LLM-ready dataset
df = pd.read_csv("llm_ready_dataset.csv")

# Ensure date is a datetime object and sort chronologically per stock
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by=['stock_id', 'date'], inplace=True)

# ---------------------------------------------------------
# 6-1. Rolling Volatility & Momentum 
# ---------------------------------------------------------
vol_window = 252        # ~12 months for volatility
mom_total_window = 252  # ~12 months total
mom_skip_window = 21    # ~1 month to skip to mitigate short-term reversal
mom_calc_window = mom_total_window - mom_skip_window # 231 days

# Calculate Annualized Rolling Volatility (Daily Std Dev * sqrt(252))
df['annual_volatility'] = df.groupby('stock_id')['DlyRet'].transform(
    lambda x: x.rolling(window=vol_window, min_periods=200).std() * np.sqrt(252)
)

# Calculate 1-Month (21-day) momentum
df['momentum_1m'] = df.groupby('stock_id')['DlyRet'].transform(
    lambda x: (1 + x).rolling(window=mom_skip_window, min_periods=10).apply(np.prod, raw=True) - 1
)

# Calculate 12M Momentum excluding the most recent month (t-12m to t-1m)
df['momentum_12m_ex_1m'] = df.groupby('stock_id')['DlyRet'].transform(
    lambda x: (1 + x).shift(mom_skip_window).rolling(window=mom_calc_window, min_periods=200).apply(np.prod, raw=True) - 1
)


# ---------------------------------------------------------
# 6-2. Fundamental Ratios
# ---------------------------------------------------------
# Accruals = (Net Income - Operating Cash Flow) / Total Assets
# Uses oancfq (single-quarter) to match niq (single-quarter); see Section 2 derivation.
df['accruals'] = np.where(
    df['atq'] != 0, 
    (df['niq'] - df['oancfq']) / df['atq'], 
    np.nan
)

# Return on Assets (ROA) = Net Income / Total Assets
df['roa'] = np.where(
    df['atq'] != 0, 
    df['niq'] / df['atq'], 
    np.nan
)

# Return on Equity (ROE) = Net Income / Total Shareholders' Equities
df['roe'] = np.where(
    (df['atq'] - df['ltq']) != 0, 
    df['niq'] / (df['atq'] - df['ltq']), 
    np.nan
)

# Quality of Earnings (QOE) = Operating Cash Flow / Net Income
# Uses oancfq (single-quarter) to match niq (single-quarter); see Section 2 derivation.
df['qoe'] = np.where(
    df['niq'] != 0, 
    df['oancfq'] / df['niq'], 
    np.nan
)


# ---------------------------------------------------------
# 6-3. Final Cleanup
# ---------------------------------------------------------
# Forward-fill any newly created NaNs within the same stock_id
fund_cols = ['accruals', 'roa', 'roe', 'qoe']
df[fund_cols] = df.groupby('stock_id')[fund_cols].ffill()

# Drop rows where we don't have enough data to calculate our new rolling metrics
df.dropna(subset=['annual_volatility', 'momentum_12m_ex_1m', 'accruals', 'roa', 'roe', 'qoe'], inplace=True)

# Save the fully augmented dataset
df.to_csv("llm_ready_dataset_augmented.csv", index=False)