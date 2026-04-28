# Data: Raw Inputs

The two raw input files consumed by `data_preprocessing.py` are **not
committed** to this repository. Both are extracts from WRDS-licensed
databases (CRSP and Compustat) which are not redistributable under
standard WRDS terms of use.

This file documents the exact pull specifications so anyone with WRDS
access can regenerate the raw files identically and run the full
pipeline end-to-end.

---

## File 1: `stock_daily_2017_2025.csv`

**Source:** CRSP — Stock / Security Files — *CRSP Daily Stock File* (`crsp.dsf` or the equivalent web query interface).

**Universe:** Current S&P 500 constituents (point-in-time membership snapshot taken at project start; no dynamic index reconstitution).

**Date range:** 2017-01-01 through 2025-12-31, daily frequency.

**Columns retrieved (minimum needed):**

| Column | Description |
|---|---|
| `Ticker` | Ticker symbol; renamed to `tic` during preprocessing |
| `DlyCalDt` | Calendar date (`YYYY-MM-DD`); renamed to `date` |
| `DlyPrc` | Daily closing price |
| `DlyRet` | Daily return (used for momentum, volatility, and performance) |
| `DlyRetx` | Daily return excluding dividends |
| `PERMNO`, `PERMCO`, `HdrCUSIP` | Identifiers; dropped during anonymization (Section 5-2 of `data_preprocessing.py`) |

**Filters applied during preprocessing:** rows with missing `DlyPrc`, `DlyRet`, or `DlyRetx` are dropped. Daily returns are winsorized at the 1st and 99th percentiles to reduce the impact of corporate-action artifacts.

---

## File 2: `fundamentals_q_2017_2025.csv`

**Source:** Compustat — North America — *Fundamentals Quarterly* (`comp.fundq`).

**Universe:** Same S&P 500 constituents as File 1, joined via ticker.

**Date range:** 2017-01-01 through 2025-12-31, quarterly frequency, indexed by reporting date `rdq`.

**Columns retrieved:**

| Column | Description |
|---|---|
| `GVKEY` | Compustat firm identifier; dropped during anonymization |
| `tic` | Ticker; the merge key with File 1 |
| `datadate` | Fiscal period end date (`DD/MM/YYYY` in source) |
| `rdq` | Report date — the date data became publicly known. Used as the time key for the point-in-time merge that prevents look-ahead bias. |
| `fyearq`, `fqtr`, `datacqtr` | Fiscal year, fiscal quarter, calendar quarter labels |
| `atq` | Total assets, quarterly |
| `ltq` | Total liabilities, quarterly |
| `niq` | Net income, quarterly |
| `epspxq` | Earnings per share excluding extraordinary items |
| `oancfy` | Operating cash flow, **year-to-date** |
| `ivncfy` | Investing cash flow, **year-to-date** |
| `fincfy` | Financing cash flow, **year-to-date** |

**Note on cash-flow items:** Compustat reports `oancfy`, `ivncfy`, `fincfy` on a *year-to-date* basis (Q1 = Q1 flow, Q2 = Q1+Q2, etc.), while `niq` is already quarterly. `data_preprocessing.py` Section 2 derives single-quarter flows (`oancfq`, `ivncfq`, `fincfq`) so that ratios such as Quality of Earnings (`oancfq / niq`) and accruals (`(niq − oancfq) / atq`) align temporally.

**Filters applied during preprocessing:** rows with missing `rdq`, `atq`, or `niq` are dropped. Duplicate quarter rows (occasionally produced by Compustat formatting codes) are de-duplicated on `(tic, datadate)`, keeping the latest `rdq`.

---

## Reproducing the raw pull

The simplest path is via the WRDS web query interface:

1. Log in to WRDS with a Cornell (or other institutional) account.
2. **CRSP → Stock / Security Files → Daily Stock File** with the universe, date range, and columns above. Save as `stock_daily_2017_2025.csv` in the repository root.
3. **Compustat → North America → Fundamentals Quarterly** with the universe, date range, and columns above. Save as `fundamentals_q_2017_2025.csv` in the repository root.

Alternatively, if you prefer SQL via the WRDS Cloud / `wrds` Python package, the date and ticker filters and column projections above translate directly to a `SELECT` against `crsp.dsf` and `comp.fundq` joined to a constituent list.

Once both files are in place, run:

```bash
python source\ codes/data_preprocessing.py
python source\ codes/data_pipeline.py
```

The resulting `annual_universe_cleaned.csv` is the entry point consumed by both `baseline_engine.py` and `llm_engine.py`.

---

## Why the raw files are not committed

WRDS subscriber agreements restrict redistribution of underlying data to
non-subscribers. Posting the raw CSVs publicly — even as part of a
course-project replication package — falls outside what the standard
agreement permits. The pipeline is structured so that everything
downstream of the two raw inputs is deterministic and fully scripted,
and the LLM-arm-specific outputs (`audit_prompts/`, `audit_responses/`)
are committed in full so that the LLM behavior is reproducible without
re-querying the model.
