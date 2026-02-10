import os
import sys
import time
import math
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from web3 import Web3

try:
    import openpyxl  # noqa: F401
except Exception:
    raise SystemExit(
        "Missing dependency: openpyxl\n"
        "Install with: python -m pip install openpyxl\n"
        "Tip: activate your .venv first."
    )

# =============================
# Configuration
# =============================

load_dotenv()

INFURA_API_KEY = os.getenv("INFURA_API_KEY") or os.getenv("api_key")
if not INFURA_API_KEY:
    raise Exception("Missing INFURA_API_KEY (or legacy api_key) in .env")

RPCS = [
    f"https://mainnet.infura.io/v3/{INFURA_API_KEY}",
    "https://eth.llamarpc.com",
]

VBILL_TOKEN_ADDRESS = "0x2255718832bc9fd3be1caf75084f4803da14ff01"
ISSUER_CONTRACT_ADDRESS = "0x22afdb66dc56be3a81285d953124bda8020dcb88"
NULL_ADDRESS = "0x0000000000000000000000000000000000000000"

ISSUER_CONTRACT_CREATION_BLOCK = 22468524
BULK_ISSUANCE = "0xb28d07c3"  # interest mint method selector

TEST_MODE = "--test" in sys.argv
TEST_SAMPLE_SIZE = 10
INCREMENTAL_MODE = "--incremental" in sys.argv
LAST_BLOCK_FILE = "vbill_last_processed_block.txt"

OUTPUT_FILENAME = "nav_volatility_analysis_test.xlsx" if TEST_MODE else "nav_volatility_analysis.xlsx"

DEFAULT_BATCH_SIZE = 10_000
MIN_BATCH_SIZE = 1_000
LLAMA_MAX_BLOCK_RANGE = 1_000

# =============================
# ABI
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

# =============================
# RPC management
# =============================

_current_rpc_idx = 0
w3 = None
_last_rotate_ts = 0.0
ROTATE_COOLDOWN_SEC = 10


def fmt_seconds(seconds: float) -> str:
    if seconds is None or seconds < 0:
        return "n/a"
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


def connect_web3() -> Web3:
    global w3
    rpc = RPCS[_current_rpc_idx]
    candidate = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 30}))
    if not candidate.is_connected():
        raise Exception(f"Failed to connect to RPC: {rpc}")
    w3 = candidate
    return w3


def current_rpc() -> str:
    return RPCS[_current_rpc_idx]


def current_rpc_is_llama() -> bool:
    return "llamarpc" in current_rpc().lower()


def effective_batch_size(base_batch_size: int) -> int:
    if current_rpc_is_llama():
        return min(base_batch_size, LLAMA_MAX_BLOCK_RANGE)
    return base_batch_size


def rotate_rpc(reason: str = "") -> None:
    global _current_rpc_idx, _last_rotate_ts, w3
    now = time.time()

    if now - _last_rotate_ts < ROTATE_COOLDOWN_SEC:
        return

    start_idx = _current_rpc_idx
    for step in range(1, len(RPCS) + 1):
        idx = (start_idx + step) % len(RPCS)
        rpc = RPCS[idx]
        candidate = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 30}))
        if candidate.is_connected():
            _current_rpc_idx = idx
            w3 = candidate
            _last_rotate_ts = now
            msg = f"Switching RPC to {rpc}"
            if reason:
                msg += f" ({reason})"
            print(f"  {msg}")
            return

    raise Exception("No reachable RPC endpoint available")


def is_rate_limit_error(msg: str) -> bool:
    s = msg.lower()
    return "429" in s or "too many requests" in s or "rate limit" in s


def is_range_too_large_error(msg: str) -> bool:
    s = msg.lower()
    return (
        "range is too large" in s
        or "max is 1k blocks" in s
        or "more than 10000 results" in s
        or "10000" in s
    )


def is_transient_error(msg: str) -> bool:
    s = msg.lower()
    transient_terms = [
        "429",
        "too many requests",
        "rate limit",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "connection reset",
        "remote end closed connection",
        "bad gateway",
        "gateway timeout",
        "service unavailable",
    ]
    return any(term in s for term in transient_terms)


def get_logs_resilient(params: dict, max_retries: int = 8):
    for attempt in range(max_retries):
        try:
            return w3.eth.get_logs(params)
        except Exception as e:
            msg = str(e).lower()

            if is_range_too_large_error(msg):
                raise

            if (
                "unauthorized" in msg
                or "authenticate your request" in msg
                or "invalid api key" in msg
                or "forbidden" in msg
            ):
                rotate_rpc("auth error on get_logs")
                wait = min(20, 2 ** attempt)
                print(f"  Auth error. Sleeping {wait}s; retrying...")
                time.sleep(wait)
                continue

            if "400 client error" in msg or "bad request" in msg:
                if attempt >= 1:
                    rotate_rpc("persistent 400 on get_logs")
                wait = min(20, 2 ** attempt)
                print(f"  400/BadRequest. Sleeping {wait}s; retrying...")
                time.sleep(wait)
                continue

            if is_transient_error(msg):
                if is_rate_limit_error(msg):
                    rotate_rpc("rate limited on get_logs")
                wait = min(45, 2 ** attempt)
                print(f"  Transient error. Sleeping {wait}s; retrying...")
                time.sleep(wait)
                continue

            raise

    raise Exception("Max retries exceeded in get_logs_resilient")


def get_tx_resilient(tx_hash, max_retries: int = 6):
    for attempt in range(max_retries):
        try:
            return w3.eth.get_transaction(tx_hash)
        except Exception as e:
            msg = str(e)
            if is_transient_error(msg):
                if is_rate_limit_error(msg):
                    rotate_rpc("rate limited on get_transaction")
                time.sleep(min(20, 2 ** attempt))
                continue
            raise
    raise Exception(f"Failed to fetch tx: {tx_hash.hex() if hasattr(tx_hash, 'hex') else tx_hash}")


def get_block_timestamp_cached(block_number: int, cache: dict) -> int:
    if block_number in cache:
        return cache[block_number]

    for attempt in range(6):
        try:
            ts = w3.eth.get_block(block_number)["timestamp"]
            cache[block_number] = ts
            return ts
        except Exception as e:
            msg = str(e)
            if is_transient_error(msg):
                if is_rate_limit_error(msg):
                    rotate_rpc("rate limited on get_block")
                time.sleep(min(20, 2 ** attempt))
                continue
            raise
    raise Exception(f"Failed to fetch timestamp for block {block_number}")


# =============================
# Incremental helpers
# =============================

def load_from_block(default_from_block: int, latest_block: int) -> int:
    if not INCREMENTAL_MODE:
        return default_from_block

    if os.path.exists(LAST_BLOCK_FILE):
        try:
            with open(LAST_BLOCK_FILE, "r", encoding="utf-8") as f:
                saved = int(f.read().strip())
            return max(default_from_block, min(saved + 1, latest_block))
        except Exception:
            return default_from_block

    return default_from_block


def save_last_block(block_number: int) -> None:
    if not INCREMENTAL_MODE:
        return
    with open(LAST_BLOCK_FILE, "w", encoding="utf-8") as f:
        f.write(str(block_number))


# =============================
# Metrics
# =============================

def add_performance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects numeric Daily_Yield_% in percent units.
    Adds:
      - daily_return (decimal)
      - APY 7D / 30D / 90D
      - APY Since Inception
      - Sharpe Ratio
    """
    out = df.copy()

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    out["Daily_Yield_%"] = pd.to_numeric(out["Daily_Yield_%"], errors="coerce")
    out["daily_return"] = out["Daily_Yield_%"] / 100.0

    for window in (7, 30, 90):
        col = f"APY {window}D"
        out[col] = (
            (1.0 + out["daily_return"])
            .rolling(window=window, min_periods=window)
            .apply(lambda x: (x.prod() ** (365.0 / window)) - 1.0, raw=False)
        ) * 100.0

    inception_apy = pd.NA
    if len(out) >= 2:
        first_date = out["Date"].iloc[0].date()
        last_date = out["Date"].iloc[-1].date()
        days_elapsed = (last_date - first_date).days

        cumulative = (1.0 + out["daily_return"].fillna(0.0)).prod()
        if days_elapsed >= 7 and cumulative > 0:
            inception_apy = (cumulative ** (365.0 / days_elapsed) - 1.0) * 100.0

    out["APY Since Inception"] = inception_apy

    sharpe = pd.NA
    dr = out["daily_return"].dropna()
    if len(dr) >= 7:
        std_daily = dr.std()
        if std_daily and std_daily > 0:
            sharpe = (dr.mean() / std_daily) * (365.0 ** 0.5)

    out["Sharpe Ratio"] = sharpe
    return out


def latest_non_null(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.iloc[-1] if len(s) > 0 else None


# =============================
# Main
# =============================

overall_start_ts = time.time()

connect_web3()
print(f"Connected to Ethereum mainnet via RPC: {current_rpc()}")

latest_block = w3.eth.block_number
print(f"Latest block: {latest_block}")

# contracts
vbill_token_full = w3.eth.contract(
    address=Web3.to_checksum_address(VBILL_TOKEN_ADDRESS),
    abi=ERC20_ABI + [TRANSFER_EVENT_ABI],
)
vbill_token = w3.eth.contract(
    address=Web3.to_checksum_address(VBILL_TOKEN_ADDRESS),
    abi=[TRANSFER_EVENT_ABI],
)

# decimals
try:
    VBILL_DECIMALS = vbill_token_full.functions.decimals().call()
    print(f"VBILL token decimals: {VBILL_DECIMALS}")
except Exception as e:
    print(f"Warning: could not query VBILL decimals, defaulting to 6. Error: {e}")
    VBILL_DECIMALS = 6

from_block = load_from_block(ISSUER_CONTRACT_CREATION_BLOCK, latest_block)

if TEST_MODE and not INCREMENTAL_MODE:
    # roughly 1 month
    to_block = min(from_block + 216_000, latest_block)
else:
    to_block = latest_block

if from_block > to_block:
    print("Nothing new to fetch (from_block > to_block).")
    sys.exit(0)

total_block_span = to_block - from_block + 1

print("\nStep 1: Getting all VBILL mint events (from null address)")
print(f"Querying blocks {from_block} -> {to_block} (span: {total_block_span:,} blocks)")
if TEST_MODE:
    print("TEST MODE enabled")

transfer_event_signature = Web3.keccak(text="Transfer(address,address,uint256)")
null_address_topic = Web3.to_hex(Web3.to_bytes(hexstr=NULL_ADDRESS).rjust(32, b"\x00"))

batch_size = DEFAULT_BATCH_SIZE
current_from = from_block
mint_events = []
processed_blocks = 0
batch_counter = 0

while current_from <= to_block:
    active_batch = effective_batch_size(batch_size)
    current_to = min(current_from + active_batch - 1, to_block)

    params = {
        "fromBlock": current_from,
        "toBlock": current_to,
        "address": Web3.to_checksum_address(VBILL_TOKEN_ADDRESS),
        "topics": [transfer_event_signature, null_address_topic],  # from = null
    }

    try:
        logs_mint = get_logs_resilient(params)
    except Exception as e:
        msg = str(e)
        if is_range_too_large_error(msg):
            new_batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
            if new_batch_size == batch_size:
                raise Exception(
                    f"Cannot reduce batch size further (already {batch_size}) but still hitting limits."
                ) from e
            print(f"  Range/result limit in {current_from}-{current_to}; batch {batch_size} -> {new_batch_size}")
            batch_size = new_batch_size
            continue
        print(f"  Error querying blocks {current_from}-{current_to}: {e}")
        print("  Retrying same range after 10s...")
        time.sleep(10)
        continue

    for log in logs_mint:
        try:
            decoded = vbill_token.events.Transfer().process_log(log)
            mint_events.append(
                {
                    "value": decoded["args"]["value"],
                    "block_number": log["blockNumber"],
                    "tx_hash": log["transactionHash"],
                }
            )
        except Exception:
            continue

    batch_counter += 1
    step_blocks = current_to - current_from + 1
    processed_blocks += step_blocks

    print(f"  Blocks {current_from}-{current_to}: Found {len(logs_mint)} mint events (total: {len(mint_events)})")

    elapsed = time.time() - overall_start_ts
    progress = processed_blocks / total_block_span if total_block_span > 0 else 1.0
    eta = (elapsed / progress) - elapsed if progress > 0 else None
    expected_total = (elapsed / progress) if progress > 0 else None

    if batch_counter % 5 == 0 or current_to == to_block:
        print(
            f"    Progress: {progress*100:.2f}% | Elapsed: {fmt_seconds(elapsed)} | "
            f"ETA: {fmt_seconds(eta)} | Expected total: {fmt_seconds(expected_total)} | RPC: {current_rpc()}"
        )

    current_from = current_to + 1

print(f"\nTotal mint events found: {len(mint_events)}")

# Step 2: classify mints
print("\nStep 2: Classifying mint events by transaction type...")

issuer_checksum = Web3.to_checksum_address(ISSUER_CONTRACT_ADDRESS)
bulk_issuance_id = BULK_ISSUANCE[2:].lower() if BULK_ISSUANCE.startswith("0x") else BULK_ISSUANCE.lower()

for i, event in enumerate(mint_events):
    try:
        tx = get_tx_resilient(event["tx_hash"])
        is_interest = False

        tx_to = tx["to"]
        if tx_to and str(tx_to).lower() == issuer_checksum.lower():
            tx_input = tx["input"]
            if isinstance(tx_input, bytes):
                method_id = tx_input[:4].hex().lower() if len(tx_input) >= 4 else ""
            else:
                method_id = tx_input[2:10].lower() if isinstance(tx_input, str) and len(tx_input) >= 10 else ""

            if method_id == bulk_issuance_id:
                is_interest = True

        event["is_interest"] = is_interest

    except Exception:
        event["is_interest"] = False

    if (i + 1) % 200 == 0:
        print(f"  Classified {i + 1}/{len(mint_events)} events...")

interest_count = sum(1 for e in mint_events if e.get("is_interest", False))
print(f"  Classified {interest_count} as interest (bulkIssuance), {len(mint_events) - interest_count} as supply increase")

# Step 3: burns
print("\nStep 3: Getting burn events (to null address)...")

batch_size = DEFAULT_BATCH_SIZE
current_from = from_block
burn_events = []
processed_blocks_burn = 0
batch_counter_burn = 0

while current_from <= to_block:
    active_batch = effective_batch_size(batch_size)
    current_to = min(current_from + active_batch - 1, to_block)

    params = {
        "fromBlock": current_from,
        "toBlock": current_to,
        "address": Web3.to_checksum_address(VBILL_TOKEN_ADDRESS),
        "topics": [transfer_event_signature, None, null_address_topic],  # to = null
    }

    try:
        logs_burn = get_logs_resilient(params)
    except Exception as e:
        msg = str(e)
        if is_range_too_large_error(msg):
            new_batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
            if new_batch_size == batch_size:
                raise Exception(
                    f"Cannot reduce burn batch size further (already {batch_size}) but still hitting limits."
                ) from e
            print(f"  Range/result limit in {current_from}-{current_to}; batch {batch_size} -> {new_batch_size}")
            batch_size = new_batch_size
            continue
        print(f"  Error querying burn blocks {current_from}-{current_to}: {e}")
        print("  Retrying same range after 10s...")
        time.sleep(10)
        continue

    for log in logs_burn:
        try:
            decoded = vbill_token.events.Transfer().process_log(log)
            burn_events.append(
                {
                    "value": decoded["args"]["value"],
                    "block_number": log["blockNumber"],
                    "tx_hash": log["transactionHash"],
                }
            )
        except Exception:
            continue

    batch_counter_burn += 1
    step_blocks = current_to - current_from + 1
    processed_blocks_burn += step_blocks

    print(f"  Blocks {current_from}-{current_to}: Found {len(logs_burn)} burn events (total: {len(burn_events)})")

    if batch_counter_burn % 5 == 0 or current_to == to_block:
        elapsed = time.time() - overall_start_ts
        progress = processed_blocks_burn / total_block_span if total_block_span > 0 else 1.0
        eta = (elapsed / progress) - elapsed if progress > 0 else None
        expected_total = (elapsed / progress) if progress > 0 else None
        print(
            f"    Progress (burn scan): {progress*100:.2f}% | Elapsed: {fmt_seconds(elapsed)} | "
            f"ETA: {fmt_seconds(eta)} | Expected total: {fmt_seconds(expected_total)} | RPC: {current_rpc()}"
        )

    current_from = current_to + 1

print(f"\nTotal burn events found: {len(burn_events)}")

# Step 4: aggregate daily
print("\nStep 4: Grouping by date and calculating daily supply changes...")

all_events = []
for event in mint_events:
    all_events.append(
        {
            "type": "mint",
            "is_interest": event.get("is_interest", False),
            "value": event["value"],
            "block_number": event["block_number"],
        }
    )
for event in burn_events:
    all_events.append(
        {
            "type": "burn",
            "is_interest": False,
            "value": event["value"],
            "block_number": event["block_number"],
        }
    )

all_events.sort(key=lambda x: x["block_number"])

daily_data = {}
current_supply = 0  # token base units
block_ts_cache = {}

for event in all_events:
    try:
        bn = event["block_number"]
        ts = get_block_timestamp_cached(bn, block_ts_cache)
        dt = datetime.fromtimestamp(ts)
        date = dt.date()

        if date not in daily_data:
            daily_data[date] = {
                "interest_minted": 0,
                "supply_minted": 0,
                "burned": 0,
                "blocks": [],
                "timestamps": [],
            }

        if event["type"] == "mint":
            if event["is_interest"]:
                daily_data[date]["interest_minted"] += event["value"]
            else:
                daily_data[date]["supply_minted"] += event["value"]
        else:
            daily_data[date]["burned"] += event["value"]

        daily_data[date]["blocks"].append(bn)
        daily_data[date]["timestamps"].append(dt)

    except Exception:
        continue

print("\nCalculating daily series...")

nav_data = []
for date in sorted(daily_data.keys()):
    data = daily_data[date]

    interest_minted = data["interest_minted"]
    supply_minted = data["supply_minted"]
    burned = data["burned"]

    net_supply_change = interest_minted + supply_minted - burned
    previous_supply = current_supply
    current_supply += net_supply_change

    if previous_supply > 0:
        daily_yield_pct = (interest_minted / previous_supply) * 100.0 if interest_minted > 0 else 0.0
    else:
        daily_yield_pct = None

    nav_data.append(
        {
            "Date": date,
            "DateTime": max(data["timestamps"]),
            "Block": max(data["blocks"]),
            "Interest_Minted": interest_minted / (10 ** VBILL_DECIMALS),
            "Supply_Minted": supply_minted / (10 ** VBILL_DECIMALS),
            "Burned": burned / (10 ** VBILL_DECIMALS),
            "Net_Supply_Change": net_supply_change / (10 ** VBILL_DECIMALS),
            "Supply_Before": previous_supply / (10 ** VBILL_DECIMALS),
            "Supply_After": current_supply / (10 ** VBILL_DECIMALS),
            "Daily_Yield_%": daily_yield_pct,
            "NAV": 1.0,
        }
    )

df_analysis = pd.DataFrame(nav_data)
if not df_analysis.empty:
    df_analysis = add_performance_metrics(df_analysis)

if TEST_MODE and not df_analysis.empty:
    df_analysis = df_analysis.head(TEST_SAMPLE_SIZE).copy()

print(f"Processed {len(df_analysis)} days of data")

# Final display frame (formatted strings for Excel parity)
final_rows = []
for _, row in df_analysis.iterrows():
    final_rows.append(
        {
            "Date": pd.to_datetime(row["Date"]).strftime("%m/%d/%Y"),
            "NAV/Share": 1.0,
            "Daily % change": f"{row['Daily_Yield_%']:.6f}%" if pd.notna(row.get("Daily_Yield_%")) else "",
            "APY 7D": f"{row['APY 7D']:.5f}%" if pd.notna(row.get("APY 7D")) else "",
            "APY 30D": f"{row['APY 30D']:.5f}%" if pd.notna(row.get("APY 30D")) else "",
            "APY 90D": f"{row['APY 90D']:.5f}%" if pd.notna(row.get("APY 90D")) else "",
            "APY Since Inception": f"{row['APY Since Inception']:.5f}%" if pd.notna(row.get("APY Since Inception")) else "",
            "Sharpe Ratio": f"{row['Sharpe Ratio']:.4f}" if pd.notna(row.get("Sharpe Ratio")) else "",
            "Interest_Minted": round(float(row["Interest_Minted"]), 2),
            "Supply_Minted": round(float(row["Supply_Minted"]), 2),
            "Burned": round(float(row["Burned"]), 2),
            "Net_Supply_Change": round(float(row["Net_Supply_Change"]), 2),
            "Supply": round(float(row["Supply_After"]), 2),
        }
    )

df_vbill = pd.DataFrame(final_rows)

# preserve other sheets
existing_sheets = {}
if os.path.exists(OUTPUT_FILENAME):
    try:
        existing_sheets = pd.read_excel(OUTPUT_FILENAME, sheet_name=None, engine="openpyxl")
        print(f"Found existing {OUTPUT_FILENAME} with {len(existing_sheets)} sheet(s)")
    except Exception as e:
        print(f"Could not read existing file, continuing with new file: {e}")

print(f"\nExporting to {OUTPUT_FILENAME}...")
with pd.ExcelWriter(OUTPUT_FILENAME, engine="openpyxl") as writer:
    df_vbill.to_excel(writer, sheet_name="VBILL", index=False)
    print(f"  Created/Updated sheet 'VBILL' with {len(df_vbill)} rows")

    for sheet_name, df in existing_sheets.items():
        if sheet_name != "VBILL":
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Preserved sheet '{sheet_name}' with {len(df)} rows")

print(f"\nSuccessfully exported VBILL analysis to {OUTPUT_FILENAME}")

# save last block (incremental)
if INCREMENTAL_MODE:
    save_last_block(to_block)
    print(f"Saved last processed block: {to_block} -> {LAST_BLOCK_FILE}")

# =============================
# Summary
# =============================

print("\n" + "=" * 110)
print("VBILL NAV TRACKING SUMMARY")
print("=" * 110)

print(f"\nTotal days tracked: {len(df_analysis)}")
print(f"Total mint events: {len(mint_events)}")
print(f"Total burn events: {len(burn_events)}")

if not df_analysis.empty:
    total_interest = pd.to_numeric(df_analysis["Interest_Minted"], errors="coerce").fillna(0).sum()
    total_supply_minted = pd.to_numeric(df_analysis["Supply_Minted"], errors="coerce").fillna(0).sum()
    total_burned = pd.to_numeric(df_analysis["Burned"], errors="coerce").fillna(0).sum()
    final_supply = pd.to_numeric(df_analysis["Supply_After"], errors="coerce").dropna().iloc[-1]

    print(f"Total Interest Minted (bulkIssuance): {total_interest:,.2f}")
    print(f"Total Supply Minted (other): {total_supply_minted:,.2f}")
    print(f"Total Burned: {total_burned:,.2f}")
    print(f"Final Supply: {final_supply:,.2f}")

    dr = pd.to_numeric(df_analysis["daily_return"], errors="coerce").dropna()

    if len(dr) > 0:
        daily_std = dr.std()  # decimal
        annual_vol = daily_std * math.sqrt(365) if pd.notna(daily_std) else float("nan")
        max_loss = (dr.min() * 100.0) if len(dr) > 0 else None

        apy7 = latest_non_null(df_analysis["APY 7D"]) if "APY 7D" in df_analysis.columns else None
        apy30 = latest_non_null(df_analysis["APY 30D"]) if "APY 30D" in df_analysis.columns else None
        apy90 = latest_non_null(df_analysis["APY 90D"]) if "APY 90D" in df_analysis.columns else None
        apy_inc = latest_non_null(df_analysis["APY Since Inception"]) if "APY Since Inception" in df_analysis.columns else None
        sharpe = latest_non_null(df_analysis["Sharpe Ratio"]) if "Sharpe Ratio" in df_analysis.columns else None

        print(f"\nMax Loss: {max_loss:.5f}%" if max_loss is not None else "\nMax Loss: n/a")
        print(f"Daily Std Dev: {daily_std * 100.0:.8f}%")
        print(f"Annualised Volatility: {annual_vol * 100.0:.4f}%")
        print(f"Latest APY 7D: {apy7:.3f}%" if apy7 is not None else "Latest APY 7D: n/a")
        print(f"Latest APY 30D: {apy30:.3f}%" if apy30 is not None else "Latest APY 30D: n/a")
        print(f"Latest APY 90D: {apy90:.3f}%" if apy90 is not None else "Latest APY 90D: n/a")
        print(f"APY Since Inception: {apy_inc:.3f}%" if apy_inc is not None else "APY Since Inception: n/a")
        print(f"Sharpe Ratio: {sharpe:.4f}" if sharpe is not None else "Sharpe Ratio: n/a")

total_elapsed = time.time() - overall_start_ts
print(f"\nDone. Total runtime: {fmt_seconds(total_elapsed)}")
