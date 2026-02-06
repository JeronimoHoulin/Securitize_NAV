# Securitize NAV Updates Analysis

This repository contains scripts for analyzing Securitize vault NAV (Net Asset Value) data from the Ethereum mainnet:

1. **`nav_updates.py`** - Fetches NAV updates for multiple vaults (VBILL, STAC, ACRED, HLSCOPE) from the Redstone multifeed adapter contract
2. **`vbill_nav_tracker.py`** - Tracks VBILL token supply changes and calculates daily interest/yield from issuance contract transactions

## Setup

### 1. Install Dependencies

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install -r requirements.txt
```

Both methods work identically. `uv` is faster and recommended if you have it installed.

### 2. Environment Variables

Create a `.env` file in the project root directory with the following variable:

- `api_key`: Your Infura API key for connecting to Ethereum mainnet

See `example.env` for a template.

### 3. Run the Script

#### Basic Usage
```bash
python nav_updates.py
```

This will query from block 20419584 (when the Redstone multifeed adapter was created) to the latest block for NAV updates and generate `nav_volatility_analysis.xlsx`.

#### Test Mode
```bash
python nav_updates.py --test
```

Limits the output to the first 10 samples per vault and generates `nav_volatility_analysis_test.xlsx`. Useful for testing and quick validation.

#### Combine Flags
```bash
python nav_updates.py --test
```

You can combine flags as needed.

## Output

The script generates an Excel file (`nav_volatility_analysis.xlsx` or `nav_volatility_analysis_test.xlsx` in test mode) with:
- One sheet per vault (VBILL, STAC, ACRED, HLSCOPE)
- Columns: Date, NAV/Share, Daily % change
- Summary statistics printed to console

---

## VBILL NAV Tracker (`vbill_nav_tracker.py`)

This script tracks VBILL token supply changes by monitoring ERC-20 Transfer events and calculates daily interest/yield from issuance contract transactions.

### Purpose

The script analyzes VBILL token mints and burns to:
- Track daily interest payments (from `bulkIssuance` function calls)
- Separate interest-generating mints from other supply increases
- Calculate daily yield percentages based on interest payments
- Monitor total supply changes over time

### Usage

#### Basic Usage
```bash
uv run vbill_nav_tracker.py
```

This will query from block 22468524 (when the issuer contract was created) to the latest block and generate `nav_volatility_analysis.xlsx` with a VBILL sheet.

#### Test Mode
```bash
uv run vbill_nav_tracker.py --test
```

Limits the query to approximately 1 month of blocks (~216,000 blocks) and generates `nav_volatility_analysis_test.xlsx`. Useful for testing and quick validation.

### How It Works

The script follows a 4-step process:

1. **Get All VBILL Mint Events**: Queries all ERC-20 Transfer events where `from` address is the null address (0x0000...), indicating token minting
2. **Classify Mint Events**: For each mint event, checks the transaction that created it:
   - If the transaction calls `bulkIssuance` (0xb28d07c3) on the issuer contract → counts as **interest**
   - Otherwise (e.g., `bulkRegisterAndIssuance`) → counts as **supply increase** (not interest)
3. **Get Burn Events**: Queries all Transfer events where `to` address is the null address, indicating token burning (supply decrease)
4. **Calculate Daily Aggregates**: Groups all events by date and calculates:
   - Daily interest minted (from `bulkIssuance` only)
   - Daily supply minted (other mints)
   - Daily burns
   - Daily yield percentage (interest / previous supply)
   - Total supply changes

### Output

The script generates an Excel file (`nav_volatility_analysis.xlsx` or `nav_volatility_analysis_test.xlsx` in test mode) with a VBILL sheet containing:

- **Date**: Date of the transaction
- **NAV/Share**: Always 1.0 (VBILL is a rebasing token with fixed NAV)
- **Daily % change**: Daily yield percentage calculated from interest mints only
- **Interest_Minted**: VBILL minted from `bulkIssuance` (interest payments)
- **Supply_Minted**: VBILL minted from other functions (supply increases)
- **Burned**: VBILL burned (supply decreases)
- **Net_Supply_Change**: Net change in supply for the day
- **Supply**: Total supply after the day's changes

The script also prints summary statistics including:
- Total days tracked
- Total mint/burn events processed
- Total interest minted vs supply minted
- Average, max, and min daily yield percentages

### Key Features

- **Accurate Interest Tracking**: Only counts `bulkIssuance` transactions as interest, not `bulkRegisterAndIssuance`
- **Daily Aggregation**: Groups events by date for clean daily reporting
- **Supply Tracking**: Maintains running supply totals to calculate accurate yield percentages
- **Batch Processing**: Queries events in batches to handle large block ranges efficiently
- **Error Handling**: Gracefully handles rate limits and continues processing

### Contract Details

- **VBILL Token**: `0x2255718832bc9fd3be1caf75084f4803da14ff01`
- **Issuer Contract**: `0x22afdb66dc56be3a81285d953124bda8020dcb88`
- **Contract Creation Block**: 22468524
- **Interest Function**: `bulkIssuance` (0xb28d07c3)

### Notes

- The script queries from block 22468524 (issuer contract creation) onwards
- VBILL uses 6 decimals (not 18)
- NAV is always $1.00 for rebasing tokens - the yield is reflected in supply increases
- Daily yield is calculated as: `(interest_minted / previous_supply) * 100`
- The script preserves existing Excel sheets when updating the file

---

## How It Works (nav_updates.py)

1. Connects to Ethereum mainnet via Infura
2. Queries `ValueUpdate` events from the Redstone multifeed adapter contract starting from block 20419584 (when the adapter was created)
3. Filters events for specific oracle IDs (VBILL, STAC, ACRED, HLSCOPE)
4. Groups NAV updates by date and calculates daily percentage changes
5. Exports results to Excel with separate sheets for each vault

**Note**: The script only queries from block 20419584 onwards since the Redstone multifeed adapter was created at that block. NAV updates do not exist before this block.
