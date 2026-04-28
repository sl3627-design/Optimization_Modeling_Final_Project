import pandas as pd
import numpy as np

def calculate_optimization_inputs(df_daily, portfolio_candidates,
                                   lookback_days=252, min_obs=60):
    """
    Calculates annualized expected returns (mu) and covariance matrices (Sigma)
    for the selected portfolio candidates at each rebalance date.

    - lookback_days: target trailing window in trading days.
    - min_obs: minimum non-NaN observations required per stock. Stocks with
               fewer are dropped from that date's portfolio.
    """
    opt_inputs = {}
    df_daily = df_daily.copy()
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    rebalance_dates = portfolio_candidates['date'].unique()

    for current_date in rebalance_dates:
        current_candidates = portfolio_candidates[
            portfolio_candidates['date'] == current_date
        ]['stock_id'].tolist()

        history_df = df_daily[
            (df_daily['date'] < current_date) &
            (df_daily['stock_id'].isin(current_candidates))
        ].drop_duplicates(subset=['date', 'stock_id'], keep='first')

        returns_pivot = history_df.pivot(
            index='date', columns='stock_id', values='DlyRet'
        )

        returns_pivot = history_df.pivot(
            index='date', columns='stock_id', values='DlyRet'
        )
        
        # Apply the lookback cap FIRST
        returns_pivot = returns_pivot.tail(lookback_days)
        
        # Then trim leading NaNs within the window. This handles both
        # (a) newly-listed stocks that started mid-window, and
        # (b) stocks that had a trading gap and resumed mid-window.
        first_valid = returns_pivot.apply(lambda c: c.first_valid_index()).dropna()
        if first_valid.empty:
            print(f"  [{current_date.date()}] No stocks have any history. Skipping.")
            continue
        window_start = first_valid.max()
        returns_pivot = returns_pivot.loc[window_start:]
        
        # Now drop stocks with fewer than min_obs observations
        obs_counts = returns_pivot.count()
        insufficient = obs_counts[obs_counts < min_obs].index.tolist()
        if insufficient:
            print(f"  [{current_date.date()}] Dropping {len(insufficient)} stocks "
                  f"with <{min_obs} obs: {insufficient[:5]}"
                  f"{'...' if len(insufficient) > 5 else ''}")
            returns_pivot = returns_pivot.drop(columns=insufficient)
            current_candidates = [t for t in current_candidates if t not in insufficient]
        
        if len(current_candidates) < 2:
            print(f"  [{current_date.date()}] <2 investable stocks. Skipping.")
            continue
        
        # ffill internal gaps (individual missing trading days within active window)
        returns_pivot = returns_pivot.ffill()

        # 5. Any residual NaN now is a genuine problem
        if returns_pivot.isna().any().any():
            missing = returns_pivot.columns[returns_pivot.isna().any()].tolist()
            raise ValueError(
                f"Residual NaN for {missing} at {current_date} "
                f"after trimming and filtering."
            )

        returns_pivot = returns_pivot.reindex(columns=current_candidates)

        mu = returns_pivot.mean().values * 252
        cov_matrix = returns_pivot.cov().values * 252

        opt_inputs[current_date] = {
            'tickers': current_candidates,
            'mu': mu,
            'cov': cov_matrix,
        }

    return opt_inputs