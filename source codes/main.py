import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import your custom modules
from input_generator import calculate_optimization_inputs
from mvo import optimize_mvo
from rbo import calculate_risk_budgets, get_deterministic_risk_buckets, optimize_rbo
from control_case import control_case
from arch.bootstrap import StationaryBootstrap


import pandas as pd

# =============================================================================
# PRIORITY 1: Evaluation Metric Helper Functions
# =============================================================================

def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown from a daily return series."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    return drawdown.min()


def downside_deviation(returns: pd.Series, threshold: float = 0.0, af: int = 252) -> float:
    """Annualized downside deviation below a daily threshold (default 0)."""
    excess = returns - threshold
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    return np.sqrt((downside ** 2).mean()) * np.sqrt(af)


def average_turnover(weight_history, ticker_history):
    """
    weight_history: list of np.arrays
    ticker_history: list of lists of ticker strings
    """
    if len(weight_history) < 2:
        return 0.0
    turnovers = []
    for i in range(1, len(weight_history)):
        old = dict(zip(ticker_history[i-1], weight_history[i-1]))
        new = dict(zip(ticker_history[i],   weight_history[i]))
        all_tickers = set(old) | set(new)
        diff = sum(abs(new.get(t, 0.0) - old.get(t, 0.0)) for t in all_tickers)
        turnovers.append(diff / 2)
    return float(np.mean(turnovers))


def plot_metrics_comparison(metrics_dict: dict, output_title: str) -> None:
    """
    Bar chart comparing numeric metrics across strategies.
    metrics_dict: {strategy_name: {metric_name: float_value}}
    """
    numeric_metrics = ['Ann. Return', 'Ann. Volatility', 'Sharpe Ratio',
                       'Sortino Ratio', 'Max Drawdown', 'Mean HHI']

    strategies = list(metrics_dict.keys())
    n_metrics = len(numeric_metrics)
    x = np.arange(n_metrics)
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, strategy in enumerate(strategies):
        values = [metrics_dict[strategy].get(m, 0) for m in numeric_metrics]
        bars = ax.bar(x + i * width, values, width, label=strategy, alpha=0.85)
        ax.bar_label(bars, fmt='%.3f', fontsize=7, padding=2)

    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(numeric_metrics, rotation=15, ha='right', fontsize=10)
    ax.set_title(f'Strategy Metrics Comparison: {output_title}', fontsize=13)
    ax.set_ylabel('Value', fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fname = output_title.replace(' ', '_').lower() + '_metrics.png'
    plt.savefig(fname)
    print(f"→ Metrics bar chart saved as '{fname}'")


def plot_hhi_over_time(hhi_records: dict, rebalance_dates: list, output_title: str) -> None:
    """Line chart of HHI at each rebalance date per strategy."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for strategy, values in hhi_records.items():
        dates = rebalance_dates[:len(values)]
        ax.plot(dates, values, marker='o', linewidth=1.5, label=strategy)

    ax.set_title(f'Weight Concentration (HHI) Over Time: {output_title}', fontsize=13)
    ax.set_xlabel('Rebalance Date', fontsize=11)
    ax.set_ylabel('HHI', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = output_title.replace(' ', '_').lower() + '_hhi.png'
    plt.savefig(fname)
    print(f"→ HHI chart saved as '{fname}'")


def plot_bootstrap_results(comparisons_results: list, output_title: str = "") -> None:
    """
    Forest plot of bootstrap Sharpe differences with 95% CIs.
    comparisons_results: list of (label, result_dict)
    """
    labels = [r[0] for r in comparisons_results]
    diffs  = [r[1]['observed_diff'] for r in comparisons_results]
    lowers = [r[1]['ci_lower'] for r in comparisons_results]
    uppers = [r[1]['ci_upper'] for r in comparisons_results]
    pvals  = [r[1]['p_value'] for r in comparisons_results]

    y = np.arange(len(labels))
    xerr_low  = [d - l for d, l in zip(diffs, lowers)]
    xerr_high = [u - d for d, u in zip(diffs, uppers)]

    fig, ax = plt.subplots(figsize=(12, 0.9 * len(labels) + 2))
    colors = ['#d62728' if p < 0.05 else '#1f77b4' for p in pvals]

    ax.errorbar(diffs, y,
                xerr=[xerr_low, xerr_high],
                fmt='none', ecolor='gray', elinewidth=1.5, capsize=5)
    ax.scatter(diffs, y, color=colors, zorder=5, s=80)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)

    # Annotate p-values
    for i, (d, p) in enumerate(zip(diffs, pvals)):
        sig = " (*)" if p < 0.05 else ""
        ax.text(max(uppers) + 0.02, i,
                f"p={p:.3f}{sig}", va='center', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Sharpe Difference (A − B)', fontsize=11)
    ax.set_title(f'Bootstrap Sharpe Differences (95% CI){" — " + output_title if output_title else ""}',
                 fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fname = ('bootstrap_sharpe' + ('_' + output_title.replace(' ', '_').lower() if output_title else '')) + '.png'
    plt.savefig(fname)
    print(f"→ Bootstrap forest plot saved as '{fname}'")


# =============================================================================
# PRIORITY 2: Bootstrap Sharpe Difference
# =============================================================================

def bootstrap_sharpe_diff(r1, r2, n_boot=10_000, af=252, seed=42, block_size=5):
    r1_arr = r1.values
    r2_arr = r2.values
    
    def sharpe(arr):
        mu = arr.mean() * af
        sigma = arr.std() * np.sqrt(af)
        return mu / sigma if sigma > 0 else 0.0
    
    observed = sharpe(r1_arr) - sharpe(r2_arr)
    
    # Stationary bootstrap preserves serial dependence
    data = np.column_stack([r1_arr, r2_arr])
    bs = StationaryBootstrap(block_size, data, seed=seed)
    
    diffs = np.empty(n_boot)
    for k, sample in enumerate(bs.bootstrap(n_boot)):
        resampled = sample[0][0]  # shape (n, 2)
        diffs[k] = sharpe(resampled[:, 0]) - sharpe(resampled[:, 1])
    
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    p_value = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    
    return {
        'observed_diff': observed,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': float(p_value)
    }


def print_bootstrap_results(label: str, result: dict) -> None:
    sig = "(*)" if result['p_value'] < 0.05 else ""
    print(
        f"  {label:<45} "
        f"diff={result['observed_diff']:+.3f}  "
        f"95% CI=[{result['ci_lower']:+.3f}, {result['ci_upper']:+.3f}]  "
        f"p={result['p_value']:.3f} {sig}"
    )


# =============================================================================
# Core Backtest
# =============================================================================
ANNUAL_RF = 0.025          # 2.5% average 3-month T-bill yield over 2019-2025
DAILY_RF  = ANNUAL_RF / 252


def run_backtest(candidates_filename: str, output_title: str) -> pd.DataFrame:
    print(f"\n{'=' * 50}")
    print(f"Executing Backtest: {output_title}")
    print(f"{'=' * 50}")

    print("Loading data...")
    df_daily = pd.read_csv("optimization_dataset.csv")   # or whatever your daily file is
    portfolio_candidates = pd.read_csv(candidates_filename)
    
    # Convert dates FIRST, before any filtering or comparisons
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    portfolio_candidates['date'] = pd.to_datetime(portfolio_candidates['date'])
    
    # Now the filter works
    min_history_days = 252
    data_start = df_daily['date'].min()
    portfolio_candidates = portfolio_candidates[
        portfolio_candidates['date'] >= data_start + pd.Timedelta(days=int(min_history_days * 1.5))
    ].copy()
    
    print(f"→ Filtered rebalance dates. Remaining: "
          f"{portfolio_candidates['date'].nunique()} "
          f"({sorted(portfolio_candidates['date'].unique())[0].date()} "
          f"to {sorted(portfolio_candidates['date'].unique())[-1].date()})")

    # ------------------------------------------------------------------
    # Detection Step: Baseline vs LLM Risk Buckets
    # ------------------------------------------------------------------
    use_llm_buckets = 'Risk_Bucket' in portfolio_candidates.columns
    if use_llm_buckets:
        print("→ Detected 'Risk_Bucket' column. Using LLM-judged risk buckets for RBO.")
    else:
        print("→ No 'Risk_Bucket' column. Using deterministic volatility-based buckets for RBO.")

    print("Generating optimization inputs (mu, Sigma)...")
    opt_data = calculate_optimization_inputs(df_daily, portfolio_candidates)
    rebalance_dates = sorted(list(opt_data.keys()))

    portfolio_daily_returns = {'Control (1/N)': [], 'MVO': [], 'RBO': []}
    hhi_records    = {'Control (1/N)': [], 'MVO': [], 'RBO': []}
    weight_history = {'Control (1/N)': [], 'MVO': [], 'RBO': []}
    ticker_history = {'Control (1/N)': [], 'MVO': [], 'RBO': []}   
    dates_list = []

    if use_llm_buckets:
        # Sidecar: same stock selection (LLM), deterministic volatility buckets
        portfolio_daily_returns['RBO_detbuckets'] = []
        hhi_records['RBO_detbuckets']   = []
        weight_history['RBO_detbuckets'] = []
        ticker_history['RBO_detbuckets'] = []

    print("Executing rolling optimizations and calculating out-of-sample returns...")
    for i, current_date in enumerate(rebalance_dates):

        next_date = (rebalance_dates[i + 1]
                     if i < len(rebalance_dates) - 1
                     else df_daily['date'].max())

        tickers = opt_data[current_date]['tickers']
        mu      = opt_data[current_date]['mu']
        cov     = opt_data[current_date]['cov']
        vols    = np.sqrt(np.diag(cov))

        # Weights
        w_control = control_case(mu)
        w_mvo     = optimize_mvo(mu, cov, risk_aversion=2.0)

        if use_llm_buckets:
            # LLM buckets (the main one)
            current_df = portfolio_candidates[portfolio_candidates['date'] == current_date]
            bucket_map = dict(zip(current_df['stock_id'], current_df['Risk_Bucket']))
            buckets    = np.array([bucket_map[t] for t in tickers])
            b     = calculate_risk_budgets(buckets)
            w_rbo = optimize_rbo(cov, b)
        
            # Sidecar: deterministic buckets on the same LLM stock selection
            det_buckets = get_deterministic_risk_buckets(pd.Series(vols))
            b_det       = calculate_risk_budgets(det_buckets)
            w_rbo_det   = optimize_rbo(cov, b_det)
        else:
            buckets = get_deterministic_risk_buckets(pd.Series(vols))
            b       = calculate_risk_budgets(buckets)
            w_rbo   = optimize_rbo(cov, b)

        # Record HHI
        hhi_records['Control (1/N)'].append(np.sum(w_control ** 2))
        hhi_records['MVO'].append(np.sum(w_mvo ** 2))
        hhi_records['RBO'].append(np.sum(w_rbo ** 2))

        # Record weights for turnover
        weight_history['Control (1/N)'].append(w_control)
        weight_history['MVO'].append(w_mvo)
        weight_history['RBO'].append(w_rbo)
        
        # Record tickers alongside weights (same 'tickers' for all three strategies
        # since they share the candidate pool)
        ticker_history['Control (1/N)'].append(tickers)
        ticker_history['MVO'].append(tickers)
        ticker_history['RBO'].append(tickers)
        
        # Sidecar recording
        if use_llm_buckets:
            hhi_records['RBO_detbuckets'].append(np.sum(w_rbo_det ** 2))
            weight_history['RBO_detbuckets'].append(w_rbo_det)
            ticker_history['RBO_detbuckets'].append(tickers)
        
        # Forward returns
        mask = (df_daily['date'] > current_date) & (df_daily['date'] <= next_date)
        forward_df = df_daily[mask & df_daily['stock_id'].isin(tickers)]

        if forward_df.empty:
            continue

        forward_df    = forward_df.drop_duplicates(subset=['date', 'stock_id'], keep='first')
        forward_pivot = forward_df.pivot(index='date', columns='stock_id', values='DlyRet')
        forward_pivot = forward_pivot.reindex(columns=tickers)
        if forward_pivot.isna().any().any():
            missing_ct = forward_pivot.isna().sum().sum()
            missing_stocks = forward_pivot.columns[forward_pivot.isna().any()].tolist()
            print(f"  Warning: {missing_ct} NaN returns for {missing_stocks} in hold window "
                  f"{current_date.date()} to {next_date.date()}. Filling with 0 (cash equivalent).")
        forward_pivot = forward_pivot.fillna(0)

        portfolio_daily_returns['Control (1/N)'].extend(forward_pivot.dot(w_control).tolist())
        portfolio_daily_returns['MVO'].extend(forward_pivot.dot(w_mvo).tolist())
        portfolio_daily_returns['RBO'].extend(forward_pivot.dot(w_rbo).tolist())
        
        if use_llm_buckets:
            portfolio_daily_returns['RBO_detbuckets'].extend(forward_pivot.dot(w_rbo_det).tolist())
        
        dates_list.extend(forward_pivot.index.tolist())

    df_results = pd.DataFrame(portfolio_daily_returns, index=dates_list)

    # ------------------------------------------------------------------
    # PRIORITY 1: Full Evaluation Metrics
    # ------------------------------------------------------------------
    print("\n--- Portfolio Evaluation Metrics ---")
    AF = 252
    cum_returns  = (1 + df_results).cumprod() - 1
    total_return = cum_returns.iloc[-1]
    ann_return   = df_results.mean() * AF
    ann_vol      = df_results.std() * np.sqrt(AF)
    
    excess_returns = df_results - DAILY_RF
    excess_ann_return = excess_returns.mean() * AF
    sharpe_ratio      = excess_ann_return / ann_vol
    
    # Sortino uses downside deviation below the RF threshold (not 0)
    dd = df_results.apply(lambda s: downside_deviation(s, threshold=DAILY_RF))
    sortino = excess_ann_return / dd
    
    mdd = df_results.apply(max_drawdown)

    # Turnover
    turnover_row = {s: f"{average_turnover(weight_history[s], ticker_history[s]):.2%}"
                for s in weight_history}

    # HHI summary
    hhi_mean = {s: np.mean(v) for s, v in hhi_records.items()}

    metrics = pd.DataFrame({
        'Total Return':    total_return.map('{:.2%}'.format),
        'Ann. Return':     ann_return.map('{:.2%}'.format),
        'Ann. Volatility': ann_vol.map('{:.2%}'.format),
        'Sharpe Ratio':    sharpe_ratio.map('{:.3f}'.format),
        'Sortino Ratio':   sortino.map('{:.3f}'.format),
        'Max Drawdown':    mdd.map('{:.2%}'.format),
        'Avg Turnover':    pd.Series(turnover_row),
        'Mean HHI':        pd.Series(hhi_mean).map('{:.4f}'.format),
    })
    print(metrics.T)   # Transpose so strategies are columns — easier to read

    print("\n--- Weight Concentration (Herfindahl-Hirschman Index) ---")
    print(f"  Theoretical minimum (1/N for N={len(tickers)}): {1/len(tickers):.4f}")
    for strategy, hhi_list in hhi_records.items():
        print(f"  {strategy}: mean={np.mean(hhi_list):.4f}, "
              f"max={np.max(hhi_list):.4f}, "
              f"min={np.min(hhi_list):.4f}")
    
    raw_metrics = {}
    for strategy in df_results.columns:
        raw_metrics[strategy] = {
            'Ann. Return':     float(ann_return[strategy]),
            'Ann. Volatility': float(ann_vol[strategy]),
            'Sharpe Ratio':    float(sharpe_ratio[strategy]),
            'Sortino Ratio':   float(sortino[strategy]),
            'Max Drawdown':    float(mdd[strategy]),      # negative number
            'Mean HHI':        hhi_mean[strategy],
        }

    plot_metrics_comparison(raw_metrics, output_title)
    plot_hhi_over_time(hhi_records, rebalance_dates, output_title)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    for col in cum_returns.columns:
        plt.plot(cum_returns.index, cum_returns[col], label=col, linewidth=1.5)

    plt.title(f'Out-of-Sample Cumulative Returns: {output_title}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    file_name = output_title.replace(' ', '_').lower() + '.png'
    plt.savefig(file_name)
    print(f"→ Plot saved as '{file_name}'")

    return df_results


# =============================================================================
# Main: Run Both Backtests, Then Bootstrap Validation
# =============================================================================

if __name__ == "__main__":

    print("Starting Baseline Evaluation...")
    baseline_results = run_backtest("baseline_candidates.csv", "Deterministic Baseline")

    print("\nStarting LLM Evaluation...")
    llm_results = run_backtest("llm_candidates.csv", "LLM Augmented Model")

    # ------------------------------------------------------------------
    # PRIORITY 2: Bootstrap Sharpe Validation (cross-model comparisons)
    # Align indices so both DataFrames cover the same dates
    # ------------------------------------------------------------------
    print(f"\n{'=' * 50}")
    print("Bootstrap Sharpe Ratio Difference Tests (10,000 resamples)")
    print("Format: diff = Sharpe(A) - Sharpe(B),  (*) = p < 0.05")
    print(f"{'=' * 50}")

    common_idx = baseline_results.index.intersection(llm_results.index)
    b = baseline_results.loc[common_idx]
    l = llm_results.loc[common_idx]

    comparisons = [
    # Signal quality: do LLM's picks dominate vol-based picks for a naive investor?
    ("Baseline 1/N  vs  LLM 1/N     [selection quality, naive investor]",
     b['Control (1/N)'], l['Control (1/N)']),

    # Does LLM selection help a sophisticated MVO investor?
    ("Baseline MVO  vs  LLM MVO     [selection quality, MVO investor]",
     b['MVO'], l['MVO']),

    # Does LLM selection help an RBO investor?
    ("Baseline RBO  vs  LLM RBO     [selection + bucket effect, combined]",
     b['RBO'], l['RBO']),

    # Within LLM: is RBO better than naive 1/N?
    ("LLM RBO       vs  LLM 1/N     [RBO value-add on LLM stocks]",
     l['RBO'], l['Control (1/N)']),

    # ISOLATED bucket effect: same LLM stocks, LLM buckets vs deterministic buckets
    ("LLM RBO (LLM buckets) vs LLM RBO (det buckets)  [bucket quality, isolated]",
     l['RBO'], l['RBO_detbuckets']),
    ]
    bootstrap_plot_data = []
    for label, r1, r2 in comparisons:
        result = bootstrap_sharpe_diff(r1, r2)
        print_bootstrap_results(label, result)
        bootstrap_plot_data.append((label, result))
    plot_bootstrap_results(bootstrap_plot_data)

    print("\nNote: Wide CIs and high p-values with annual rebalancing are expected")
    print("given the small number of rebalance events. Report point estimates")
    print("alongside intervals — non-significance is itself an informative finding.")