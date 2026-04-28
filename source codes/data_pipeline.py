import pandas as pd

# Load data and ensure datetime format
df = pd.read_csv("llm_ready_dataset_augmented.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['stock_id', 'date'])

# --- Sections 1 & 2: Momentum and Annual Dates ---
fundamental_cols = ['roa', 'roe', 'qoe', 'accruals']
quarterly_data = df[['stock_id', 'datacqtr'] + fundamental_cols].drop_duplicates()
quarterly_data = quarterly_data.sort_values(['stock_id', 'datacqtr'])

for col in fundamental_cols:
    quarterly_data[f'delta_{col}'] = quarterly_data.groupby('stock_id')[col].diff()

df = df.merge(
    quarterly_data[['stock_id', 'datacqtr'] + [f'delta_{col}' for col in fundamental_cols]],
    on=['stock_id', 'datacqtr'], how='left'
)
df = df.dropna(subset=[f'delta_{col}' for col in fundamental_cols])

df['year'] = df['date'].dt.to_period('Y')
market_year_ends = df.groupby('year')['date'].transform('max')
df_annual = df[df['date'] == market_year_ends].copy()
df_annual = df_annual.drop(columns=['year'])

# --- Section 3: Standardize and Score ---
def z_score(series):
    std_val = series.std()
    if pd.isna(std_val) or std_val < 1e-8:
        std_val = 1e-8
    return (series - series.mean()) / std_val

cols_to_standardize = ['momentum_12m_ex_1m', 'delta_roa', 'delta_roe', 'delta_qoe', 'delta_accruals']
for col in cols_to_standardize:
    df_annual[f'z_{col}'] = df_annual.groupby('date')[col].transform(z_score)

df_annual['FM_Score'] = (df_annual['z_delta_roa'] + df_annual['z_delta_roe'] + df_annual['z_delta_qoe'] - df_annual['z_delta_accruals'])
df_annual['z_FM_Score'] = df_annual.groupby('date')['FM_Score'].transform(z_score)
df_annual['Twin_Score'] = df_annual['z_momentum_12m_ex_1m'] + df_annual['z_FM_Score']

# --- Section 3.5: Clean the Universe ---
df_annual = df_annual.sort_values(by=['date', 'Twin_Score'], ascending=[True, False])
df_annual = df_annual.drop_duplicates(subset=['date', 'atq', 'niq'], keep='first')
df_annual = df_annual[df_annual['date'] >= '2018-11-30']

# STOP HERE. Save the entire scored universe for both engines to look at.
df_annual.to_csv("annual_universe_cleaned.csv", index=False)
print("Data pipeline complete. Universe saved to annual_universe_cleaned.csv")