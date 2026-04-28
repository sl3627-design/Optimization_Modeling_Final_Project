import os

import pandas as pd


# ============================================================================
# Snapshot helper
# ============================================================================

def get_llm_snapshot(dataset, target_date):
    """Isolate the cross-section of stocks for a specific rebalance date."""
    dataset = dataset.copy()
    dataset['date'] = pd.to_datetime(dataset['date'])
    target_date = pd.to_datetime(target_date)
    return dataset[dataset['date'] == target_date].copy()


# ============================================================================
# Anonymization guard
# ============================================================================

_FORBIDDEN_COLUMNS = {'tic', 'GVKEY', 'PERMNO', 'PERMCO', 'HdrCUSIP', 'CUSIP', 'CUSIP9'}


def _assert_anonymized(df):
    leaked = _FORBIDDEN_COLUMNS & set(df.columns)
    if leaked:
        raise ValueError(
            f"Anonymization leak: forbidden columns {leaked} present. "
            f"Strip these upstream."
        )


# ============================================================================
# Shared helpers
# ============================================================================

def _format_date(target_date):
    return (
        target_date.strftime('%Y-%m-%d')
        if isinstance(target_date, pd.Timestamp) else str(target_date)[:10]
    )


def _validate_n_trials(n_trials):
    if not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError(f"n_trials must be a positive integer (got {n_trials!r}).")


# ============================================================================
# Prompt 1: Selection (single- or multi-trial via n_trials)
# ============================================================================

def generate_llm_selection_prompt(
    cross_section_df,
    target_date,
    k_selections=40,
    n_trials=20,
):
    """
    Build the stock-selection prompt.

    Parameters
    ----------
    cross_section_df : DataFrame
        Anonymized cross-section for a single rebalance date.
    target_date : str or pd.Timestamp
        Rebalance date.
    k_selections : int, default 40
        Number of stocks in the final selection.
    n_trials : int, default 1
        If 1, the LLM performs one holistic selection.
        If > 1, the LLM is instructed to run `n_trials` independent selection
        trials internally, aggregate by selection frequency, and return a
        single consensus top-`k_selections` CSV. The output schema is
        identical in both cases.
    """
    _assert_anonymized(cross_section_df)
    _validate_n_trials(n_trials)

    formatted_date = _format_date(target_date)
    multi_trial = n_trials > 1

    # --- Header ------------------------------------------------------------
    if multi_trial:
        header = (
            f"You are an expert quantitative portfolio manager. The current date is {formatted_date}.\n"
            f"Your task is to select a consensus top-{k_selections} stock portfolio to hold "
            f"for the next 12 months, derived from {n_trials} independent selection trials.\n\n"
        )
    else:
        header = (
            f"You are an expert quantitative portfolio manager. The current date is {formatted_date}.\n"
            f"Your task is to select the top {k_selections} stocks to hold for the next 12 months.\n\n"
        )

    # --- Analytical mandate (identical in both modes) ----------------------
    mandate = (
        "### YOUR ANALYTICAL MANDATE ###\n"
        "Holistically evaluate the provided universe. Balance stable fundamental growth, "
        "earnings quality, momentum, and manageable risk.\n\n"
        "- Profitability (ROA & ROE): Higher is generally better.\n"
        "- Quality of Earnings (QoE): Higher is better.\n"
        "- Accruals: Lower (or negative) is generally better.\n"
        "- 12M Momentum Score: Higher is better.\n"
        "- Annualized Volatility: Lower is preferred unless compensated by strong fundamentals.\n\n"
        "Do not apply rigid single-metric cutoffs. Look for the best composite profiles.\n\n"
    )

    # --- Multi-trial procedure (only when n_trials > 1) --------------------
    procedure = ""
    if multi_trial:
        procedure = (
            "### MULTI-TRIAL PROCEDURE ###\n"
            f"Run exactly {n_trials} independent selection trials on the universe below. "
            "Each trial must:\n"
            f"  1. Score every stock on the mandate above and pick its own top {k_selections}.\n"
            "  2. Vary meaningfully from the other trials. Achieve this by perturbing the "
            "composite in one or more of these ways per trial: shift the relative weights on "
            "profitability / QoE / accruals / momentum / volatility within defensible ranges; "
            "apply mild winsorization or rank-based transforms to different factors; break "
            "near-ties on the boundary differently; tilt slightly toward quality vs. momentum "
            "vs. low-vol. Do NOT change the sign of any mandate relationship (e.g., higher "
            "momentum is always better).\n"
            "  3. The first trial should reflect your single best composite; subsequent trials "
            "should explore nearby defensible configurations.\n\n"

            "### AGGREGATION ###\n"
            f"After the {n_trials} trials, count how often each stock was selected. "
            f"Return the {k_selections} stocks with the highest selection frequency. "
            f"Break frequency ties using the average composite score across the trials "
            f"in which the stock appeared (higher is better).\n\n"

            "### INTERNAL REASONING ###\n"
            "Do all trial scoring, aggregation, and tie-breaking internally. Do NOT include "
            "per-trial picks, scores, counts, or any reasoning in the output. The output must "
            "be only the final consensus CSV.\n\n"
        )

    # --- Universe data -----------------------------------------------------
    universe = "### UNIVERSE DATA ###\n"
    for _, row in cross_section_df.iterrows():
        def fmt(val):
            return f"{val:.4f}" if pd.notna(val) else "N/A"

        universe += f"\n#### Stock ID: {row['stock_id']} ####\n"
        universe += f"* ROA: {fmt(row.get('roa'))}\n"
        universe += f"* ROE: {fmt(row.get('roe'))}\n"
        universe += f"* QoE: {fmt(row.get('qoe'))}\n"
        universe += f"* Accruals: {fmt(row.get('accruals'))}\n"
        universe += f"* 12M Momentum (ex-1M): {fmt(row.get('momentum_12m_ex_1m'))}\n"
        universe += f"* Annualized Volatility: {fmt(row.get('annual_volatility'))}\n"

    # --- Output format -----------------------------------------------------
    if multi_trial:
        ordering_note = (
            " Order the rows by selection frequency (most frequent first); within ties, "
            "higher average composite score first."
        )
        extra_strict = (
            "- Each row contains ONLY the raw stock identifier. No features, no scores, no "
            "frequencies, no citation markers, no annotations, no brackets of any kind.\n"
            "- Output only the raw CSV inside a single fenced code block (```csv ... ```). "
            "No conversational text, no per-trial output, no explanations before or after.\n"
        )
    else:
        ordering_note = ""
        extra_strict = (
            "- Each row contains ONLY the raw stock identifier. No features, no scores, no "
            "citation markers, no annotations, no brackets of any kind.\n"
            "- Output only the raw CSV inside a single fenced code block (```csv ... ```). "
            "No conversational text, no explanations before or after.\n"
        )

    output_spec = (
        "\n### OUTPUT FORMAT ###\n"
        f"Return a CSV with a single column named `stock_id` and exactly {k_selections} rows "
        "(plus the header). Each row should contain the identifier of one selected stock "
        f"(e.g. stock_389).{ordering_note}\n\n"
        f"The CSV will be saved as: audit_responses/llm_selected_{formatted_date}.csv\n\n"
        "Example:\n"
        "stock_id\n"
        "stock_17\n"
        "stock_203\n"
        "stock_45\n"
        "...\n\n"
        "STRICT REQUIREMENTS:\n"
        "- Use the exact column name `stock_id` (lowercase with underscore).\n"
        f"- Exactly {k_selections} data rows plus one header row. No more, no fewer.\n"
        + extra_strict
    )

    return header + mandate + procedure + universe + output_spec


# ============================================================================
# Prompt 2: Risk categorization (single- or multi-trial via n_trials)
# ============================================================================

def generate_llm_risk_categorization_prompt(
    selected_df,
    target_date,
    n_trials=20,
):
    """
    Build the risk-categorization prompt. `selected_df` must contain only the
    stocks already picked by the selection prompt.

    Parameters
    ----------
    selected_df : DataFrame
        Anonymized data for the already-selected stocks.
    target_date : str or pd.Timestamp
        Rebalance date.
    n_trials : int, default 1
        If 1, the LLM performs one holistic categorization.
        If > 1, the LLM is instructed to run `n_trials` independent
        categorization trials internally and return a single consensus
        assignment per stock (majority vote; ties broken by the trial-average
        bucket, rounded to the nearest integer in {1..5}). The output schema
        is identical in both cases.
    """
    _assert_anonymized(selected_df)
    _validate_n_trials(n_trials)

    n_stocks = len(selected_df)
    if n_stocks == 0:
        raise ValueError("selected_df is empty.")

    target_per_bucket = n_stocks // 5
    formatted_date = _format_date(target_date)
    multi_trial = n_trials > 1

    # --- Header ------------------------------------------------------------
    if multi_trial:
        header = (
            f"You are an expert quantitative portfolio manager. The current date is {formatted_date}.\n"
            f"I have provided {n_stocks} stocks for a risk-budgeting portfolio. Derive a "
            f"consensus risk-bucket assignment for each stock from {n_trials} independent "
            "categorization trials.\n\n"
        )
    else:
        header = (
            f"You are an expert quantitative portfolio manager. The current date is {formatted_date}.\n"
            f"I have provided {n_stocks} stocks for a risk-budgeting portfolio. Assign each stock "
            f"to a risk bucket.\n\n"
        )

    # --- Output requirement ------------------------------------------------
    output_req = (
        "### OUTPUT REQUIREMENT ###\n"
        "Add a new column `Risk_Bucket` with integer values in {1, 2, 3, 4, 5}:\n"
        "- **1** = safest\n"
        "- **5** = riskiest\n\n"
    )

    # --- Distribution constraint ------------------------------------------
    distribution = (
        "### BUCKET DISTRIBUTION CONSTRAINT ###\n"
        f"The final distribution MUST be approximately uniform: approximately {target_per_bucket} "
        f"stocks per bucket (acceptable range: {target_per_bucket - 1} to {target_per_bucket + 1} "
        "per bucket). Prioritize relative ordering over absolute classifications. "
        "Do not under-populate Bucket 1 -- it should contain the safest stocks "
        f"in the cross-section, not only 'absolutely safe' stocks. Target {target_per_bucket} "
        "stocks in Bucket 1.\n\n"
    )

    # --- Methodology -------------------------------------------------------
    methodology = (
        "### METHODOLOGY ###\n"
        "Step 1 -- Volatility anchor. Rank by `annual_volatility` and place into quintiles. "
        "This is the starting point.\n\n"
        "Step 2 -- Fundamental adjustment (within +/- 1 bucket of the anchor):\n"
        "- Adjust upward (riskier) if low volatility hides weak fundamentals: negative "
        "profitability, weak cash flows, poor earnings quality, elevated leverage.\n"
        "- Adjust downward (safer) if elevated volatility is backed by robust fundamentals: "
        "strong profitability, high-quality earnings, clean balance sheet.\n\n"
        "After adjustments, rebalance bucket populations to restore approximate uniformity.\n\n"
    )

    # --- Multi-trial procedure (only when n_trials > 1) --------------------
    procedure = ""
    if multi_trial:
        procedure = (
            "### MULTI-TRIAL PROCEDURE ###\n"
            f"Run exactly {n_trials} independent categorization trials. Each trial must "
            "follow the methodology above and produce a bucket assignment for every stock, "
            "while varying meaningfully from the other trials. Acceptable sources of "
            "variation: weight the fundamental adjustment vs. the volatility anchor "
            "differently; vary which fundamental signals dominate (profitability, QoE, "
            "accruals, leverage, cash-flow quality); break near-quintile-boundary ties "
            "differently. Do NOT violate the distribution constraint in any trial.\n\n"

            "### AGGREGATION ###\n"
            "For each stock, derive the consensus bucket by majority vote across the "
            f"{n_trials} trials. Break ties using the trial-average bucket rounded to the "
            "nearest integer in {1..5}. After aggregation, if the consensus distribution "
            "violates the uniformity constraint, reassign the minimum number of boundary "
            "stocks (those with the closest average bucket to an adjacent bucket) to "
            "restore approximate uniformity.\n\n"

            "### INTERNAL REASONING ###\n"
            "Do all trial scoring, voting, tie-breaking, and rebalancing internally. "
            "Do NOT include per-trial assignments, vote counts, or any reasoning in the "
            "output. The output must be only the final consensus CSV.\n\n"
        )

    # --- Downstream-use context -------------------------------------------
    downstream = (
        "### DOWNSTREAM USE (context) ###\n"
        "Bucket labels map to risk budgets via b_i proportional to 1/bucket_i. Bucket 1 receives "
        "the largest share of the total risk budget at the optimum.\n\n"
    )

    # --- Input data --------------------------------------------------------
    input_data = "### INPUT DATA ###\n"
    for _, row in selected_df.iterrows():
        def fmt(val):
            return f"{val:.4f}" if pd.notna(val) else "N/A"

        input_data += f"\n#### Stock ID: {row['stock_id']} ####\n"
        input_data += f"* Annualized Volatility: {fmt(row.get('annual_volatility'))}\n"
        input_data += f"* ROA: {fmt(row.get('roa'))}\n"
        input_data += f"* ROE: {fmt(row.get('roe'))}\n"
        input_data += f"* QoE: {fmt(row.get('qoe'))}\n"
        input_data += f"* Accruals: {fmt(row.get('accruals'))}\n"
        input_data += f"* Net Income (niq): {fmt(row.get('niq'))}\n"
        input_data += f"* Operating Cash Flow (oancfq): {fmt(row.get('oancfq'))}\n"
        input_data += f"* Total Assets (atq): {fmt(row.get('atq'))}\n"
        input_data += f"* Total Liabilities (ltq): {fmt(row.get('ltq'))}\n"
        input_data += f"* FM Score: {fmt(row.get('FM_Score'))}\n"

    # --- Output format -----------------------------------------------------
    output_spec = (
        "\n### OUTPUT FORMAT ###\n"
        "Output the augmented dataset as raw CSV text inside a single fenced code block "
        "(```csv ... ```). Retain all original columns and add `Risk_Bucket` as the final "
        "column. Use exact column names as provided; keep `stock_id` lowercase with "
        "underscore. Include the header.\n\n"
        f"The CSV will be saved as: audit_responses/llm_risk_categorized_{formatted_date}.csv\n\n"
        f"Output must contain exactly {n_stocks} rows plus the header, and every stock_id "
        "from the input must appear exactly once.\n\n"
        "STRICT REQUIREMENTS:\n"
        "- No citation markers, source references, footnotes, or bracketed annotations "
        "anywhere in the output. No [cite: N] tags, no [1] markers, no square brackets at all.\n"
        "- The `stock_id` column must contain only the raw identifier (e.g. `stock_389`), "
        "not `stock_389[cite: 7]` or similar.\n"
        "- No conversational text before or after the CSV.\n"
    )

    return (
        header
        + output_req
        + distribution
        + methodology
        + procedure
        + downstream
        + input_data
        + output_spec
    )


# ============================================================================
# Prompt saving
# ============================================================================

def save_prompt(prompt_text, task, target_date, prompt_dir="audit_prompts"):
    """Save a prompt to disk with a predictable filename."""
    os.makedirs(prompt_dir, exist_ok=True)
    date_str = _format_date(target_date)
    filename = os.path.join(prompt_dir, f"prompt_{task}_{date_str}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    return filename


# ============================================================================
# Response loading and validation
# ============================================================================

def load_llm_selection(target_date, response_dir="audit_responses", k_expected=None):
    """
    Load the LLM's selection CSV and return the list of selected stock_ids.
    Works for both single-trial and multi-trial (consensus) outputs because
    the schema is identical.

    Expected filename: llm_selected_<YYYY-MM-DD>.csv
    """
    date_str = _format_date(target_date)
    path = os.path.join(response_dir, f"llm_selected_{date_str}.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"LLM selection CSV not found at {path}")

    df = pd.read_csv(path)

    if 'stock_id' not in df.columns:
        raise ValueError(f"{path} is missing the 'stock_id' column.")

    ids = df['stock_id'].tolist()
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} contains duplicate stock_ids.")

    if k_expected is not None and len(ids) != k_expected:
        raise ValueError(
            f"{path} has {len(ids)} rows, expected {k_expected}."
        )

    return ids


def load_llm_risk_buckets(target_date, expected_stock_ids, response_dir="audit_responses"):
    """
    Load the LLM's risk-categorization CSV and return buckets as a Series
    aligned to expected_stock_ids.

    Expected filename: llm_risk_categorized_<YYYY-MM-DD>.csv
    """
    date_str = _format_date(target_date)
    path = os.path.join(response_dir, f"llm_risk_categorized_{date_str}.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"LLM risk CSV not found at {path}")

    df = pd.read_csv(path)

    # Schema checks
    if 'stock_id' not in df.columns or 'Risk_Bucket' not in df.columns:
        raise ValueError(
            f"{path} must contain 'stock_id' and 'Risk_Bucket' columns. "
            f"Got: {df.columns.tolist()}"
        )

    # Universe alignment checks
    returned_ids = set(df['stock_id'])
    expected_ids = set(expected_stock_ids)

    missing = expected_ids - returned_ids
    if missing:
        raise ValueError(f"Risk CSV at {path} omits: {missing}")

    extras = returned_ids - expected_ids
    if extras:
        raise ValueError(f"Risk CSV at {path} contains unexpected stock_ids: {extras}")

    # Value checks
    if df['Risk_Bucket'].isna().any():
        na_ids = df.loc[df['Risk_Bucket'].isna(), 'stock_id'].tolist()
        raise ValueError(f"NaN Risk_Bucket for: {na_ids}")

    if not df['Risk_Bucket'].between(1, 5).all():
        bad = df.loc[~df['Risk_Bucket'].between(1, 5), ['stock_id', 'Risk_Bucket']]
        raise ValueError(f"Invalid Risk_Bucket values:\n{bad}")

    # Align to expected_stock_ids so the downstream b-vector matches
    # the order of mu and Sigma
    bucket_map = dict(zip(df['stock_id'], df['Risk_Bucket'].astype(int)))
    return pd.Series(
        [bucket_map[sid] for sid in expected_stock_ids],
        index=expected_stock_ids,
        name='Risk_Bucket'
    )