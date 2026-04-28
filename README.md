# LLM Signals vs. Deterministic Factors in Portfolio Construction: A 2×2 Backtest

A 2×2 experimental framework comparing **Mean–Variance Optimization (MVO)**
against **Risk-Budgeting Optimization (RBO)**, each driven by either an
LLM-based asset selection / risk-bucketing layer or a deterministic
factor-based baseline. All four arms are constructed from a common
S&P 500 cross-section and evaluated on identical out-of-sample windows.

| | **Asset selection: LLM** | **Asset selection: Twin-Score baseline** |
|---|---|---|
| **MVO** (long-only Markowitz) | Arm A | Arm B |
| **RBO** (risk-budgeted, LLM or 1/σ buckets) | Arm C | Arm D |

The deterministic baseline follows the **Twin Momentum** construction of
Huang, Zhang, and Zhou (2020). The MVO formulation follows
Boyd et al. (2024). The RBO formulation follows the risk-budgeting
literature surveyed in Uysal, Li, and Mulvey (2021); neural-network
signal generation is explicitly out of scope.

---

## Repository contents

```
.
├── source codes/        # Pipeline + engines + backtest driver
├── audit_prompts/       # Exact LLM prompts sent at each rebalance date
├── audit_responses/     # LLM CSV responses (selections + risk buckets)
├── asset selections/    # Final selected portfolios per arm per date
└── outputs/             # Backtest results, weights, performance tables
```

### `source codes/`

| File | Role |
|---|---|
| `data_preprocessing.py` | Cleans CRSP daily file and Compustat quarterly fundamentals; converts YTD cash-flow items to single-quarter flows; performs point-in-time `merge_asof` to prevent look-ahead bias; winsorizes daily returns at 1%/99%; anonymizes tickers to `stock_id`; computes momentum, volatility, and fundamental ratios (ROA, ROE, QoE, accruals). |
| `data_pipeline.py` | Builds the annual cross-section: computes Δ on quarterly fundamentals, z-scores momentum and fundamental factors per rebalance date, and produces the **Twin Score** = `z(momentum_12m_ex_1m) + z(FM_Score)`. Output: `annual_universe_cleaned.csv`. |
| `baseline_engine.py` | Deterministic top-K selection by Twin Score for each rebalance date. |
| `llm_engine.py` | Builds the two LLM prompts (selection and risk categorization), saves them to `audit_prompts/`, and validates / loads the LLM's CSV responses from `audit_responses/`. Includes anonymization assertions to prevent identifier leakage. |
| `main_backtest.py` | Top-level driver. Runs in three modes: `selection` (generate selection prompts), `risk` (generate risk-bucketing prompts from saved selections), and `baseline` (deterministic selection only, no LLM). |

### `audit_prompts/` and `audit_responses/`

These two folders are the audit trail of the LLM arm. Because LLM outputs
are non-deterministic and re-querying is costly, the exact prompts and
their corresponding CSV responses are committed verbatim, enabling
inspection without re-running the model.

- `audit_prompts/prompt_selection_<YYYY-MM-DD>.txt` — full prompt for the top-40 selection
- `audit_prompts/prompt_risk_<YYYY-MM-DD>.txt` — full prompt for the 5-bucket risk categorization
- `audit_responses/llm_selected_<YYYY-MM-DD>.csv` — 40 stock_ids
- `audit_responses/llm_risk_categorized_<YYYY-MM-DD>.csv` — 40 rows × {stock_id, …, Risk_Bucket ∈ {1..5}}

Rebalance dates: 2018-12-31, 2019-12-31, 2020-12-31, 2021-12-31,
2022-12-30, 2023-12-29, 2024-12-31, 2025-12-31.

---

## Reproducing the results

### Requirements

- Python ≥ 3.10
- `pandas`, `numpy`, `cvxpy`, plus a CVXPY-compatible solver (Clarabel ships with cvxpy ≥ 1.5; ECOS / SCS also work)
- WRDS access for the raw inputs (see **Data** below)

```bash
pip install pandas numpy cvxpy
```

### Data

The two raw input files are **not** in this repository. They are
licensed CRSP and Compustat extracts pulled from WRDS and cannot be
redistributed under standard WRDS terms of use. See [`data/README.md`](data/README.md)
for the exact pull specifications (date range, universe, columns) so that
anyone with WRDS access can regenerate them identically.

Once the two raw CSVs are placed at the repo root:

```bash
python source\ codes/data_preprocessing.py    # raw → cleaned → merged → winsorized → llm_ready_dataset_augmented
python source\ codes/data_pipeline.py         # → annual_universe_cleaned.csv
```

### Running the backtest

The LLM arm runs in two passes because the LLM is queried interactively
through a chat UI rather than via API:

```bash
# Pass 1 — generate selection prompts and the deterministic baseline
python source\ codes/main_backtest.py selection
# Then paste each audit_prompts/prompt_selection_<date>.txt into the LLM
# and save responses as audit_responses/llm_selected_<date>.csv

# Pass 2 — generate risk-categorization prompts from the saved selections
python source\ codes/main_backtest.py risk
# Then paste each audit_prompts/prompt_risk_<date>.txt into the LLM
# and save responses as audit_responses/llm_risk_categorized_<date>.csv

# Deterministic baseline only (no LLM)
python source\ codes/main_backtest.py baseline
```

The audit prompts and responses for the rebalance dates listed above are
already committed under `audit_prompts/` and `audit_responses/`, so the
LLM arm is reproducible **without** re-querying the model.

---

## Anonymization protocol

Identifiers are dropped before the cross-section is shown to the LLM.
The mapping `stock_id → ticker` is held out in
`ticker_anonymization_key.csv` and is **never** included in any prompt.
This prevents the LLM from leveraging name- or sector-level priors
encoded during pretraining and isolates its scoring to the numerical
factor profile alone. `llm_engine.py` enforces this with an
`_assert_anonymized` guard that raises if any of `tic`, `GVKEY`,
`PERMNO`, `PERMCO`, `HdrCUSIP`, `CUSIP`, `CUSIP9` appears in the
DataFrame passed into either prompt builder.

---

## References

- Boyd, S., Johansson, K., Kahn, R., Schiele, P., & Schmelzer, T. (2024). *Markowitz Portfolio Construction at Seventy*.
- Huang, D., Zhang, H., & Zhou, G. (2020). Twin Momentum: Fundamental Trends Matter. *SSRN Working Paper*.
- Uysal, A. S., Li, X., & Mulvey, J. M. (2021). End-to-End Risk Budgeting Portfolio Optimization with Neural Networks. *Annals of Operations Research*.

---

## Citation

If this code or the audit trail is useful in your own work:

```bibtex
@misc{lee2025_llmrbo,
  author = {Lee, Sang-Hoon},
  title  = {LLM Signals vs. Deterministic Factors in Portfolio Construction:
            A 2x2 Backtest of MVO and RBO on the S\&P 500},
  year   = {2026},
  note   = {Cornell University, Optimization Modeling Final Project},
  url    = {https://github.com/sl3627-design/Optimization_Modeling_Final_Project}
}
```
