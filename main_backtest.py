"""
Prompt generator for the LLM arm of the 2x2 backtest.

Usage:
    python main_backtest.py selection
        Generates the stock-selection prompts for all rebalance dates.
        After running this, paste each prompt into the chat UI and save
        the response as audit_responses/llm_selected_<YYYY-MM-DD>.csv

    python main_backtest.py risk
        For each rebalance date, loads the LLM's selection from
        audit_responses/, filters the snapshot to those 40 stocks, and
        generates the risk-categorization prompt. After running this,
        paste each prompt and save the response as
        audit_responses/llm_risk_categorized_<YYYY-MM-DD>.csv

    python main_backtest.py baseline
        Runs the deterministic (factor-baseline) selection only.
        No LLM interaction required.
"""

import os
import sys
import pandas as pd

from llm_engine import (
    get_llm_snapshot,
    generate_llm_selection_prompt,
    generate_llm_risk_categorization_prompt,
    save_prompt,
    load_llm_selection,
)
from baseline_engine import get_baseline_selection


# =============================================================================
# Configuration
# =============================================================================

REBALANCE_DATES = [
    '2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31',
    '2022-12-30', '2023-12-29', '2024-12-31', '2025-12-31'
]
K_SELECTIONS = 40
UNIVERSE_FILE = "annual_universe_cleaned.csv"
PROMPT_DIR = "audit_prompts"
RESPONSE_DIR = "audit_responses"


# =============================================================================
# Pass 1: stock-selection prompts
# =============================================================================

def run_selection_pass(df_annual):
    """
    For each rebalance date, generate:
      (a) the factor-baseline selection (saved as base_picks_<date>.csv by
          baseline_engine, no LLM needed), and
      (b) the LLM selection prompt (saved to audit_prompts/).
    """
    print(f"{'=' * 60}\nPASS 1: SELECTION PROMPTS\n{'=' * 60}")
    os.makedirs(PROMPT_DIR, exist_ok=True)

    for target_date in REBALANCE_DATES:
        print(f"\n--- {target_date} ---")

        # Factor baseline: fully automated, no LLM
        baseline_picks = get_baseline_selection(
            df_annual, target_date, k_selections=K_SELECTIONS
        )
        print(f"  Baseline: selected {len(baseline_picks)} stocks "
              f"(saved to base_picks_{target_date}.csv)")

        # LLM selection: generate prompt only
        snapshot = get_llm_snapshot(df_annual, target_date)
        prompt = generate_llm_selection_prompt(
            snapshot, target_date, k_selections=K_SELECTIONS
        )
        path = save_prompt(
            prompt, task='selection', target_date=target_date,
            prompt_dir=PROMPT_DIR
        )
        print(f"  LLM selection prompt: saved to {path}")

    print(f"\n{'=' * 60}")
    print("Next step: paste each audit_prompts/prompt_selection_<date>.txt")
    print(f"into the chat UI and save the CSV response to {RESPONSE_DIR}/ as:")
    print("  llm_selected_<YYYY-MM-DD>.csv")
    print("When all 8 files are saved, run: python main_backtest.py risk")
    print(f"{'=' * 60}")


# =============================================================================
# Pass 2: risk-categorization prompts
# =============================================================================

def run_risk_pass(df_annual):
    """
    For each rebalance date, load the LLM's selection CSV, filter the
    snapshot to those stocks, and generate the risk-categorization prompt.
    """
    print(f"{'=' * 60}\nPASS 2: RISK-CATEGORIZATION PROMPTS\n{'=' * 60}")
    os.makedirs(PROMPT_DIR, exist_ok=True)

    missing_selections = []

    for target_date in REBALANCE_DATES:
        print(f"\n--- {target_date} ---")

        # Load the LLM's selection from pass 1
        try:
            selected_ids = load_llm_selection(
                target_date, response_dir=RESPONSE_DIR
            )
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            missing_selections.append(target_date)
            continue

        if len(selected_ids) != K_SELECTIONS:
            print(f"  WARNING: selection for {target_date} has "
                  f"{len(selected_ids)} stocks, expected {K_SELECTIONS}. "
                  f"Proceeding with {len(selected_ids)}.")

        # Filter the snapshot to the LLM-picked stocks only
        snapshot = get_llm_snapshot(df_annual, target_date)
        selected_df = snapshot[snapshot['stock_id'].isin(selected_ids)].copy()

        # Sanity: universe alignment
        universe_ids = set(snapshot['stock_id'])
        missing_from_universe = set(selected_ids) - universe_ids
        if missing_from_universe:
            raise ValueError(
                f"LLM selection at {target_date} contains stocks not in the "
                f"universe for that date: {missing_from_universe}. "
                f"Check the selection CSV for hallucinated stock_ids."
            )

        # Preserve the LLM's selection order for consistency
        selected_df = selected_df.set_index('stock_id').loc[selected_ids].reset_index()

        # Generate the risk-categorization prompt
        prompt = generate_llm_risk_categorization_prompt(selected_df, target_date)
        path = save_prompt(
            prompt, task='risk', target_date=target_date,
            prompt_dir=PROMPT_DIR
        )
        print(f"  Risk prompt: saved to {path} ({len(selected_df)} stocks)")

    print(f"\n{'=' * 60}")
    if missing_selections:
        print(f"WARNING: no LLM selection found for {len(missing_selections)} dates:")
        for d in missing_selections:
            print(f"  - {d}")
        print("Run the selection pass for these dates before the risk pass.")
    else:
        print(f"Next step: paste each audit_prompts/prompt_risk_<date>.txt")
        print(f"into the chat UI and save the CSV response to {RESPONSE_DIR}/ as:")
        print("  llm_risk_categorized_<YYYY-MM-DD>.csv")
        print("When all 8 files are saved, run main.py to execute the backtest.")
    print(f"{'=' * 60}")


# =============================================================================
# Baseline only
# =============================================================================

def run_baseline_only(df_annual):
    """Run factor-baseline selection for all rebalance dates, no LLM."""
    print(f"{'=' * 60}\nBASELINE SELECTION ONLY\n{'=' * 60}")
    for target_date in REBALANCE_DATES:
        picks = get_baseline_selection(
            df_annual, target_date, k_selections=K_SELECTIONS
        )
        print(f"  {target_date}: {len(picks)} stocks")
    print(f"\n{'=' * 60}")
    print("After all dates complete, run merge_files.py to produce")
    print("baseline_candidates.csv and llm_candidates.csv.")
    print(f"{'=' * 60}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    # sys.argv = ['main_backtest.py', 'selection']
    sys.argv = ['main_backtest.py', 'risk']
    # sys.argv = ['main_backtest.py', 'baseline']
    if len(sys.argv) != 2 or sys.argv[1] not in {'selection', 'risk', 'baseline'}:
        print("Usage: python main_backtest.py {selection|risk|baseline}")
        print("  selection - generate LLM selection prompts (pass 1)")
        print("  risk      - generate LLM risk prompts from saved selections (pass 2)")
        print("  baseline  - run factor-baseline selection only (no LLM)")
        sys.exit(1)

    mode = sys.argv[1]

    print(f"Loading universe from {UNIVERSE_FILE}...")
    df_annual = pd.read_csv(UNIVERSE_FILE)

    if mode == 'selection':
        run_selection_pass(df_annual)
    elif mode == 'risk':
        run_risk_pass(df_annual)
    elif mode == 'baseline':
        run_baseline_only(df_annual)


if __name__ == "__main__":
    main()