import pandas as pd
import glob
import os
import re

def merge_candidate_files(file_pattern, universe_file, output_filename):
    """
    Finds all CSV files matching a pattern (e.g. llm_risk_categorized_*.csv or
    base_picks_*.csv), extracts the date from each filename, enriches each file
    with extra columns from asset_universe_cleaned.csv, merges everything into a
    master file, and sorts chronologically.
    """
    # 1. Load the full asset universe (contains all columns)
    universe_df = pd.read_csv(universe_file)
    universe_df['date'] = pd.to_datetime(universe_df['date'])
    print(f"Loaded universe file: {len(universe_df)} rows, columns: {list(universe_df.columns)}")

    # 2. Find all files matching the pattern
    all_files = sorted(glob.glob(file_pattern))
    if not all_files:
        print(f"No files found matching: {file_pattern}")
        return

    enriched_dfs = []

    for file in all_files:
        # 3. Extract date from filename (e.g., llm_risk_categorized_2021-12-31.csv or base_picks_2021-12-31.csv)
        match = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(file))
        if not match:
            print(f"  WARNING: Could not extract date from '{file}', skipping.")
            continue

        file_date = pd.to_datetime(match.group(1))
        print(f"  Processing: {os.path.basename(file)} → date: {file_date.date()}")

        # 4. Read the candidate file and assign the date
        candidate_df = pd.read_csv(file)
        candidate_df['date'] = file_date

        # 5. Get the matching universe slice for this date
        universe_slice = universe_df[universe_df['date'] == file_date]

        if universe_slice.empty:
            print(f"    WARNING: No matching rows in universe for date {file_date.date()}, keeping original columns only.")
            enriched_dfs.append(candidate_df)
            continue

        # 6. Identify extra columns in the universe not already in the candidate file
        candidate_cols = set(candidate_df.columns)
        extra_cols = [c for c in universe_slice.columns if c not in candidate_cols]
        merge_keys = ['stock_id', 'date'] if 'date' in universe_slice.columns else ['stock_id']

        # 7. Merge candidate data with the extra universe columns
        enriched = candidate_df.merge(
            universe_slice[merge_keys + extra_cols],
            on=merge_keys,
            how='left'
        )

        # 8. Drop pandas duplicate suffix columns (_x / _y) and any remaining duplicate column names
        dup_suffix_cols = [c for c in enriched.columns if c.endswith('_x') or c.endswith('_y')]
        enriched = enriched.drop(columns=dup_suffix_cols)
        enriched = enriched.loc[:, ~enriched.columns.duplicated()]

        print(f"    Merged {len(candidate_df)} rows → {len(enriched)} enriched rows (+{len(extra_cols)} columns)")
        enriched_dfs.append(enriched)

    if not enriched_dfs:
        print("No data to merge.")
        return

    # 9. Concatenate all enriched DataFrames
    master_df = pd.concat(enriched_dfs, ignore_index=True)

    # 10. Sort chronologically, then by Twin_Score descending if present
    sort_cols = ['date', 'Twin_Score'] if 'Twin_Score' in master_df.columns else ['date']
    sort_asc  = [True, False] if len(sort_cols) == 2 else [True]
    master_df = master_df.sort_values(by=sort_cols, ascending=sort_asc)

    # 11. Save
    master_df.to_csv(output_filename, index=False)
    print(f"\nSaved '{output_filename}'")
    print(f"Total rows: {len(master_df)} | Unique dates: {master_df['date'].nunique()}")
    print(f"Final columns ({len(master_df.columns)}): {list(master_df.columns)}")


# --- Execution ---
merge_candidate_files(
    file_pattern    = "llm_risk_categorized_*.csv",
    universe_file   = "annual_universe_cleaned.csv",
    output_filename = "llm_candidates.csv"
)

merge_candidate_files(
    file_pattern    = "base_picks_*.csv",
    universe_file   = "annual_universe_cleaned.csv",
    output_filename = "baseline_candidates.csv"
)