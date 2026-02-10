import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from web3 import Web3
import pandas as pd

# =============================
# Configuration
# =============================

load_dotenv()

# Env vars
# Prefer INFURA_API_KEY, fallback to legacy api_key
INFURA_API_KEY = os.getenv("INFURA_API_KEY") or os.getenv("api_key")
if not INFURA_API_KEY:
    raise Exception("Missing INFURA_API_KEY (or legacy api_key) in .env")

INFURA_BASE_URL = "https://mainnet.infura.io/v3/"

# Contracts
VBILL_TOKEN_ADDRESS = "0x2255718832bc9fd3be1caf75084f4803da14ff01"
ISSUER_CONTRACT_ADDRESS = "0x22afdb66dc56be3a81285d953124bda8020dcb88"
NULL_ADDRESS = "0x0000000000000000000000000000000000000000"

# Blocks
ISSUER_CONTRACT_CREATION_BLOCK = 22468524

# Method selectors
BULK_ISSUANCE = "0xb28d07c3"  # interest mint

# Modes
TEST_MODE = "--test" in sys.argv
OUTPUT_FILENAME = "nav_volatility_analysis_test.xlsx" if TEST_MODE else "nav_volatility_analysis.xlsx"

# Analysis window for "since date" metric
ANALYSIS_START_DATE_STR = os.getenv("ANALYSIS_START_DATE", "2025-08-21")
ANALYSIS_START_DATE = datetime.strptime(ANALYSIS_START_DATE_STR, "%Y-%m-%d").date()


def parse_arg_value(prefix: str):
    for arg in sys.argv:
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return None


# Optional CLI override: --start-date=YYYY-MM-DD
cli_start = parse_arg_value("--start-date=")
if cli_start:
    ANALYSIS_START_DATE = datetime.strptime(cli_start, "%Y-%m-%d").date()

# =============================
# Web3 setup
# =============================

infura_url = f"{INFURA_BASE_URL}{INFURA_API_KEY}"
w3 = Web3(Web3.HTTPProvider(infura_url, request_kwargs={"timeout": 60}))

if not w3.is_connected():
    raise Exception(f"Failed to connect to Infura endpoint: {infura_url}")

latest_block = w3.eth.block_number
print("Connected to Ethereum mainnet via Infura")
print(f"Latest block: {latest_block}")

# =============================
# ABI / events
# =============================

TRANSFER_EVENT_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True, "internalType": "address", "name": "from", "type": "address"},
        {"indexed": True, "internalType": "address", "name": "to", "type": "address"},
        {"indexed": False, "internalType": "uint256", "name": "value", "type": "uint256"},
    ],
    "name": "Transfer",
    "type": "event",
}

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    }
]

vbill_token_full = w3.eth.contract(
    address=Web3.to_checksum_address(VBILL_TOKEN_ADDRESS),
    abi=ERC20_ABI + [TRANSFER_EVENT_ABI],
)
vbill_token = w3.eth.contract(
    address=Web3.to_checksum_address(VBILL_TOKEN_ADDRESS),
    abi=[TRANSFER_EVENT_ABI],
)

try:
    VBILL_DECIMALS = vbill_token_full.functions.decimals().call()
    print(f"VBILL token decimals: {VBILL_DECIMALS}")
except Exception as e:
    print(f"Warning: Could not query VBILL decimals, defaulting to 6. Error: {e}")
    VBILL_DECIMALS = 6

transfer_event_signature = Web3.keccak(text="Transfer(address,address,uint256)")
null_address_topic = Web3.to_hex(Web3.to_bytes(hexstr=NULL_ADDRESS).rjust(32, b"\x00"))
issuer_checksum = Web3.to_checksum_address(ISSUER_CONTRACT_ADDRESS).lower()

# =============================
# Helpers
# =============================

def annualised_vol_from_daily_pct_std(daily_std_pct: float):
    if daily_std_pct is None:
        return None
    # daily_std_pct is in percent units (e.g. 0.01 means 0.01%)
    return daily_std_pct * (365 ** 0.5)


def latest_non_null(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.iloc[-1] if len(s) > 0 else None


def compute_since_inception_apy(df: pd.DataFrame, start_date=None):
    """
    Computes annualised APY from first row to last row, based on cumulative growth
    reconstructed from daily_return.
    If start_date is provided, uses rows from that date onward.
    """
    if df.empty:
        return None

    work = df.copy()
    work["Date_dt"] = pd.to_datetime(work["Date"], errors="coerce")

    if start_date is not None:
        work = work[work["Date_dt"].dt.date >= start_date].copy()

    work = work.sort_values("Date_dt").reset_index(drop=True)

    if len(work) < 2:
        return None

    first_date = work["Date_dt"].iloc[0].date()
    last_date = work["Date_dt"].iloc[-1].date()
    days_elapsed = (last_date - first_date).days
    if days_elapsed < 7:
        return None

    dr = pd.to_numeric(work["daily_return"], errors="coerce").fillna(0.0)
    growth = (1.0 + dr).prod()
    if growth <= 0:
        return None

    apy = (growth ** (365.0 / days_elapsed) - 1.0) * 100.0
    return apy


def rolling_apy(series_daily_return: pd.Series, window: int) -> pd.Series:
    """
    Annualised APY over rolling window from daily returns.
    """
    return (
        (1 + series_daily_return)
        .rolling(window=window, min_periods=window)
        .apply(lambda x: (x.prod() ** (365.0 / window)) - 1.0, raw=False)
    ) * 100.0


# =============================
# Range selection
# =============================

from_block = ISSUER_CONTRACT_CREATION_BLOCK
if TEST_MODE:
    # ~1 month (~216,000 blocks)
    to_block = min(from_block + 216000, latest_block)
    print(f"TEST MODE: Querying blocks {from_block} to {to_block} (~1 month)")
else:
    to_block = latest_block
    print(f"Querying blocks {from_block} to {to_block}")

# =============================
# Step 1: Mint events (from null)
# =============================

print("\nStep 1: Fetching VBILL mint events (from null address)...")

batch_size = 10_000
mint_events = []
current_from = from_block

while current_from <= to_block:
    current_to = min(current_from + batch_size - 1, to_block)
    try:
        logs_mint = w3.eth.get_logs(
            {
                "fromBlock": current_from,
                "toBlock": current_to,
                "address": Web3.to_checksum_address(VBILL_TOKEN_ADDRESS),
                "topics": [transfer_event_signature, null_address_topic],
            }
        )

        for log in logs_mint:
            try:
                decoded = vbill_token.events.Transfer().process_log(log)
                mint_events.append(
                    {
                        "value": decoded["args"]["value"],
                        "block_number": log["blockNumber"],
                        "tx_hash": log["transactionHash"].hex(),
                    }
                )
            except Exception:
                continue

        print(f"  Blocks {current_from}-{current_to}: Found {len(logs_mint)} mint events (total: {len(mint_events)})")
        current_from = current_to + 1

    except Exception as e:
        msg = str(e).lower()
        if "more than 10000 results" in msg or "10000" in msg:
            batch_size = max(1000, batch_size // 2)
            print(f"  Hit result limit. Reducing batch size to {batch_size} and retrying...")
        else:
            print(f"  Error querying {current_from}-{current_to}: {e}. Skipping range.")
            current_from = current_to + 1

print(f"\nTotal mint events found: {len(mint_events)}")

# =============================
# Step 2: Classify mints
# =============================

print("\nStep 2: Classifying mint events (interest vs non-interest)...")

bulk_issuance_id = BULK_ISSUANCE[2:].lower() if BULK_ISSUANCE.startswith("0x") else BULK_ISSUANCE.lower()

for i, event in enumerate(mint_events, start=1):
    is_interest = False
    try:
        tx = w3.eth.get_transaction(event["tx_hash"])
        tx_to = tx.get("to")
        tx_input = tx.get("input")

        if tx_to and tx_to.lower() == issuer_checksum and tx_input:
            if isinstance(tx_input, bytes):
                method_id_hex = tx_input[:4].hex() if len(tx_input) >= 4 else ""
            else:
                method_id_hex = tx_input[:10] if len(tx_input) >= 10 else ""

            method_id = method_id_hex[2:].lower() if method_id_hex.startswith("0x") else method_id_hex.lower()
            if method_id == bulk_issuance_id:
                is_interest = True

    except Exception:
        is_interest = False

    event["is_interest"] = is_interest

    if i % 100 == 0:
        print(f"  Classified {i}/{len(mint_events)} events...")

interest_count = sum(1 for e in mint_events if e.get("is_interest", False))
print(f"  Interest mints (bulkIssuance): {interest_count}")
print(f"  Non-interest mints: {len(mint_events) - interest_count}")

# =============================
# Step 3: Burn events (to null)
# =============================

print("\nStep 3: Fetching burn events (to null address)...")

burn_events = []
current_from = from_block

while current_from <= to_block:
    current_to = min(current_from + batch_size - 1, to_block)
    try:
        logs_burn = w3.eth.get_logs(
            {
                "fromBlock": current_from,
                "toBlock": current_to,
                "address": Web3.to_checksum_address(VBILL_TOKEN_ADDRESS),
                "topics": [transfer_event_signature, None, null_address_topic],
            }
        )

        for log in logs_burn:
            try:
                decoded = vbill_token.events.Transfer().process_log(log)
                burn_events.append(
                    {
                        "value": decoded["args"]["value"],
                        "block_number": log["blockNumber"],
                        "tx_hash": log["transactionHash"].hex(),
                    }
                )
            except Exception:
                continue

        print(f"  Blocks {current_from}-{current_to}: Found {len(logs_burn)} burn events (total: {len(burn_events)})")
        current_from = current_to + 1

    except Exception as e:
        msg = str(e).lower()
        if "more than 10000 results" in msg or "10000" in msg:
            batch_size = max(1000, batch_size // 2)
            print(f"  Hit result limit. Reducing batch size to {batch_size} and retrying...")
        else:
            print(f"  Error querying {current_from}-{current_to}: {e}. Skipping range.")
            current_from = current_to + 1

print(f"\nTotal burn events found: {len(burn_events)}")

# =============================
# Step 4: Build daily series
# =============================

print("\nStep 4: Aggregating per day...")

all_events = []
for e in mint_events:
    all_events.append(
        {
            "type": "mint",
            "is_interest": e.get("is_interest", False),
            "value": int(e["value"]),
            "block_number": int(e["block_number"]),
        }
    )

for e in burn_events:
    all_events.append(
        {
            "type": "burn",
            "is_interest": False,
            "value": int(e["value"]),
            "block_number": int(e["block_number"]),
        }
    )

all_events.sort(key=lambda x: x["block_number"])

daily_data = {}
block_ts_cache = {}

def get_block_ts(block_number: int):
    if block_number in block_ts_cache:
        return block_ts_cache[block_number]
    ts = w3.eth.get_block(block_number)["timestamp"]
    block_ts_cache[block_number] = ts
    return ts

for ev in all_events:
    try:
        bn = ev["block_number"]
        ts = get_block_ts(bn)
        d = datetime.fromtimestamp(ts).date()
    except Exception:
        continue

    if d not in daily_data:
        daily_data[d] = {
            "interest_minted": 0,
            "supply_minted": 0,
            "burned": 0,
            "blocks": [],
        }

    if ev["type"] == "mint":
        if ev["is_interest"]:
            daily_data[d]["interest_minted"] += ev["value"]
        else:
            daily_data[d]["supply_minted"] += ev["value"]
    else:
        daily_data[d]["burned"] += ev["value"]

    daily_data[d]["blocks"].append(bn)

sorted_dates = sorted(daily_data.keys())
current_supply = 0

rows = []
for d in sorted_dates:
    x = daily_data[d]
    interest_minted = x["interest_minted"]
    supply_minted = x["supply_minted"]
    burned = x["burned"]

    net_change = interest_minted + supply_minted - burned
    supply_before = current_supply
    supply_after = current_supply + net_change

    if supply_before > 0:
        daily_return = (interest_minted / supply_before) if interest_minted > 0 else 0.0
        daily_pct_change = daily_return * 100.0
    else:
        daily_return = None
        daily_pct_change = None

    rows.append(
        {
            "Date": d,
            "NAV/Share": 1.0,
            "Daily % change": daily_pct_change,
            "daily_return": daily_return,
            "Interest_Minted": interest_minted / (10 ** VBILL_DECIMALS),
            "Supply_Minted": supply_minted / (10 ** VBILL_DECIMALS),
            "Burned": burned / (10 ** VBILL_DECIMALS),
            "Net_Supply_Change": net_change / (10 ** VBILL_DECIMALS),
            "Supply_Before": supply_before / (10 ** VBILL_DECIMALS),
            "Supply_After": supply_after / (10 ** VBILL_DECIMALS),
            "Block": max(x["blocks"]) if x["blocks"] else None,
        }
    )

    current_supply = supply_after

df = pd.DataFrame(rows)
print(f"\nProcessed {len(df)} days of data")

if df.empty:
    print("No daily data produced. Exiting.")
    sys.exit(0)

# Rolling APYs (computed on full history)
for window in (7, 30, 90):
    col = f"APY {window}D"
    if len(df) < window:
        df[col] = pd.NA
    else:
        dr = pd.to_numeric(df["daily_return"], errors="coerce")
        df[col] = rolling_apy(dr, window)

# Since inception APY columns
apy_full = compute_since_inception_apy(df, start_date=None)
apy_since_date = compute_since_inception_apy(df, start_date=ANALYSIS_START_DATE)

df["APY Since Inception"] = apy_full
df["APY Since Inception (since analysis start)"] = apy_since_date

# Sharpe (annualised, rf = 0)
dr_valid = pd.to_numeric(df["daily_return"], errors="coerce").dropna()
sharpe = pd.NA
if len(dr_valid) >= 7:
    std_daily = dr_valid.std()
    if std_daily and std_daily > 0:
        sharpe = (dr_valid.mean() / std_daily) * (365 ** 0.5)
df["Sharpe Ratio"] = sharpe

# =============================
# Prepare Excel output
# =============================

# Keep pretty display columns
excel_df = df.copy()
excel_df["Date"] = pd.to_datetime(excel_df["Date"], errors="coerce").dt.strftime("%m/%d/%Y")
excel_df["NAV/Share"] = pd.to_numeric(excel_df["NAV/Share"], errors="coerce").round(6)

pct_cols = [
    "Daily % change",
    "APY 7D",
    "APY 30D",
    "APY 90D",
    "APY Since Inception",
    "APY Since Inception (since analysis start)",
]
for col in pct_cols:
    excel_df[col] = excel_df[col].apply(lambda x: f"{x:.5f}%" if pd.notna(x) else "")

excel_df["Sharpe Ratio"] = excel_df["Sharpe Ratio"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

# Preserve other sheets
existing_sheets = {}
if os.path.exists(OUTPUT_FILENAME):
    try:
        existing_sheets = pd.read_excel(OUTPUT_FILENAME, sheet_name=None, engine="openpyxl")
        print(f"Found existing file with {len(existing_sheets)} sheets")
    except Exception as e:
        print(f"Could not read existing workbook, will overwrite file. Error: {e}")
        existing_sheets = {}

print(f"\nExporting to {OUTPUT_FILENAME}...")
with pd.ExcelWriter(OUTPUT_FILENAME, engine="openpyxl") as writer:
    excel_df.to_excel(writer, sheet_name="VBILL", index=False)
    print(f"  Created/Updated sheet 'VBILL' with {len(excel_df)} rows")

    for sheet_name, old_df in existing_sheets.items():
        if sheet_name != "VBILL":
            old_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Preserved sheet '{sheet_name}' with {len(old_df)} rows")

print(f"\nSuccessfully exported VBILL NAV analysis to {OUTPUT_FILENAME}")

# =============================
# Summary
# =============================

print("\n" + "=" * 110)
print("VBILL NAV TRACKING SUMMARY")
print("=" * 110 + "\n")

print(f"Total days tracked: {len(df)}")
print(f"Total mint events: {len(mint_events)}")
print(f"Total burn events: {len(burn_events)}")

total_interest_minted = float(df["Interest_Minted"].sum())
total_supply_minted = float(df["Supply_Minted"].sum())
total_burned = float(df["Burned"].sum())
final_supply = float(df["Supply_After"].iloc[-1])

print(f"Total Interest Minted (bulkIssuance): {total_interest_minted:,.2f}")
print(f"Total Supply Minted (other): {total_supply_minted:,.2f}")
print(f"Total Burned: {total_burned:,.2f}")
print(f"Final Supply: {final_supply:,.2f}")

daily_changes = pd.to_numeric(df["Daily % change"], errors="coerce").dropna()
if len(daily_changes) > 0:
    max_loss = float(daily_changes.min())
    daily_std_dev = float(daily_changes.std())  # in %-points/day
    ann_vol = annualised_vol_from_daily_pct_std(daily_std_dev)  # in %-points/year
    print(f"\nMax Loss: {max_loss:.5f}%")
    print(f"Daily Std Dev: {daily_std_dev:.8f}%")
    print(f"Annualised Volatility: {ann_vol:.4f}%")
else:
    print("\nMax Loss: n/a")
    print("Daily Std Dev: n/a")
    print("Annualised Volatility: n/a")

apy7 = latest_non_null(df["APY 7D"]) if "APY 7D" in df.columns else None
apy30 = latest_non_null(df["APY 30D"]) if "APY 30D" in df.columns else None
apy90 = latest_non_null(df["APY 90D"]) if "APY 90D" in df.columns else None
apy_full_latest = latest_non_null(df["APY Since Inception"]) if "APY Since Inception" in df.columns else None
apy_since_date_latest = latest_non_null(df["APY Since Inception (since analysis start)"]) if "APY Since Inception (since analysis start)" in df.columns else None
sharpe_latest = latest_non_null(df["Sharpe Ratio"]) if "Sharpe Ratio" in df.columns else None

print(f"\nLatest APY 7D: {apy7:.3f}%" if apy7 is not None else "\nLatest APY 7D: n/a")
print(f"Latest APY 30D: {apy30:.3f}%" if apy30 is not None else "Latest APY 30D: n/a")
print(f"Latest APY 90D: {apy90:.3f}%" if apy90 is not None else "Latest APY 90D: n/a")
print(f"APY Since Inception (full history): {apy_full_latest:.3f}%" if apy_full_latest is not None else "APY Since Inception (full history): n/a")
print(
    f"APY Since Inception (since {ANALYSIS_START_DATE}): {apy_since_date_latest:.3f}%"
    if apy_since_date_latest is not None
    else f"APY Since Inception (since {ANALYSIS_START_DATE}): n/a"
)
print(f"Sharpe Ratio: {sharpe_latest:.4f}" if sharpe_latest is not None else "Sharpe Ratio: n/a")
print()
