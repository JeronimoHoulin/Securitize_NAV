# Securitize NAV Updates Analysis

This repository contains scripts for analysing Securitize vault NAV (Net Asset Value) data from Ethereum mainnet:

1. **`nav_updates.py`** - Fetches NAV updates for **STAC, ACRED, and HLSCOPE** from the Redstone multifeed adapter contract, then computes daily metrics and APY windows.
2. **`vbill_nav_tracker.py`** - Tracks VBILL token supply changes and calculates daily interest/yield from issuance contract transactions.

## Setup

### 1. Install dependencies

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install -r requirements.txt
```

Both methods work. `uv` is typically faster.

### 2. Environment variables

Create a `.env` file in the project root with:

- `INFURA_API_KEY` (preferred), or
- `api_key` (legacy fallback)

See `example.env` for a template.

### 3. Run the script (`nav_updates.py`)

#### Basic usage
```bash
python nav_updates.py
```

Queries from block `20419584` (Redstone adapter creation) to latest and generates `nav_volatility_analysis.xlsx`.

#### Test mode
```bash
python nav_updates.py --test
```

Limits output to the first 10 rows per vault and writes `nav_volatility_analysis_test.xlsx`.

#### Incremental mode
```bash
python nav_updates.py --incremental
```

Reads from `last_processed_block.txt` (if present), merges with existing Excel history, and updates only new blocks.

#### Combine flags
```bash
python nav_updates.py --incremental --test
```

---

## `nav_updates.py` output

The script generates an Excel file (`nav_volatility_analysis.xlsx`, or test variant) with:

- One sheet per vault: **STAC**, **ACRED**, **HLSCOPE**
- Columns:
  - `Date`
  - `NAV/Share`
  - `Daily % change`
  - `APY 7D`
  - `APY 30D`
  - `APY 90D`
  - `APY Since Inception`
  - `Sharpe Ratio`

It also prints a console summary per vault (days, max loss, NAV range, APYs, Sharpe).

---

## Inception anchors (important)

`nav_updates.py` supports optional anchor configuration via:

- **`inception_anchor.json`**

Expected structure:
```json
{
  "STAC": { "date": "2025-12-18", "nav": 1000.0 }
}
```

If present, anchor data is used for ‘since inception’ calculations (and any anchor-aware logic in the script).  
If missing, the script defaults to dataset-first-date behaviour.

---

## Incremental mode behaviour

When using `--incremental`:

- The script reads the start block from `last_processed_block.txt` (plus one).
- It merges new daily data with existing Excel sheets.
- It saves the latest processed block back to `last_processed_block.txt`.

If there are no new matching NAV events, it may still finish successfully without adding rows.

---

## RPC and resilience notes

`nav_updates.py` uses:

- Infura (primary, via API key)
- LlamaRPC (fallback)

The script includes retry logic, RPC rotation, and batch-size handling for transient errors/rate limits.

---

## How `nav_updates.py` works

1. Connects to Ethereum mainnet.
2. Queries `ValueUpdate` events from Redstone multifeed adapter:
   - `0xd72a6ba4a87ddb33e801b3f1c7750b2d0911fc6c`
3. Filters for configured oracle IDs:
   - STAC
   - ACRED
   - HLSCOPE
4. Builds daily NAV series (last update per day wins).
5. Computes:
   - Daily % change
   - Rolling APY (7/30/90D)
   - APY since inception
   - Sharpe ratio
6. Exports to Excel.

---

## VBILL NAV Tracker (`vbill_nav_tracker.py`)

This script tracks VBILL token supply changes by monitoring ERC-20 Transfer events and calculates daily interest/yield from issuance contract transactions.

### Purpose

The script analyses VBILL mints and burns to:

- Track daily interest payments (from `bulkIssuance` function calls)
- Separate interest-generating mints from other supply increases
- Calculate daily yield percentages based on interest payments
- Monitor total supply changes over time

### Usage

#### Basic usage
```bash
uv run vbill_nav_tracker.py
```

Queries from block `22468524` (issuer contract creation) to latest and generates `nav_volatility_analysis.xlsx` with a VBILL sheet.

#### Test mode
```bash
uv run vbill_nav_tracker.py --test
```

Limits query range (about 1 month of blocks) and generates `nav_volatility_analysis_test.xlsx`.

### How it works

1. **Get all VBILL mint events**: Transfer events with `from = 0x000...000`
2. **Classify mints** by transaction call:
   - `bulkIssuance` (`0xb28d07c3`) on issuer contract → **interest**
   - Other mint paths (for example `bulkRegisterAndIssuance`) → **supply increase**
3. **Get burn events**: Transfer events with `to = 0x000...000`
4. **Calculate daily aggregates**:
   - Interest minted
   - Supply minted
   - Burned
   - Net supply change
   - Daily yield % (`interest / previous supply`)
   - Running supply

### VBILL output columns

- `Date`
- `NAV/Share` (fixed at 1.0)
- `Daily % change`
- `Interest_Minted`
- `Supply_Minted`
- `Burned`
- `Net_Supply_Change`
- `Supply`

### Contract details

- **VBILL token**: `0x2255718832bc9fd3be1caf75084f4803da14ff01`
- **Issuer contract**: `0x22afdb66dc56be3a81285d953124bda8020dcb88`
- **Contract creation block**: `22468524`
- **Interest function**: `bulkIssuance` (`0xb28d07c3`)

### Notes

- VBILL uses 6 decimals.
- NAV is fixed at $1.00 (rebasing mechanics reflect yield via supply changes).
- The script preserves existing Excel sheets when updating files.
