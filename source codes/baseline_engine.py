import pandas as pd

def get_baseline_selection(dataset, target_date, k_selections=40):
    """
    Applies the deterministic selection rule (Section 4 of the original script).
    """
    # Ensure datetime format for safe filtering
    dataset['date'] = pd.to_datetime(dataset['date'])
    target_date = pd.to_datetime(target_date)

    # Isolate the specific rebalance date
    current_universe = dataset[dataset['date'] == target_date].copy()

    # Apply the selection
    top_picks = current_universe.sort_values(
    by=['Twin_Score', 'stock_id'], ascending=[False, True]
        ).head(k_selections)

    # Sanitize date string for use in filename (avoid colons on Windows)
    date_str = str(target_date.date())  # e.g. "2018-12-31"
    top_picks.to_csv(f"base_picks_{date_str}.csv", index=False)

    # FIX: return the actual picks list, not 0
    return top_picks['stock_id'].tolist()