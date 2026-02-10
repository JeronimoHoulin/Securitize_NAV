import os
import sys
import time
import json
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
# ALCHEMY disabled for now (free tier too restrictive for getLogs range)
# ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")

if not INFURA_API_KEY:
    raise Exception("Missing INFURA_API_KEY in .env (or legacy api_key)")

RPCS = [
    f"https://mainnet.infura.io/v3/{INFURA_API_KEY}",
    "https://eth.llamarpc.com",
]

REDSTONE_MULTIFEED_ADAPTER = "0xd72a6ba4a87ddb33e801b3f1c7750b2d0911fc6c"

ORACLE_IDS = {
    "0x535441435f46554e44414d454e54414c00000000000000000000000000000000": "STAC",
    "0x41435245445f46554e44414d454e54414c000000000000000000000000000000": "ACRED",
    "0x484c53636f70655f46554e44414d454e54414c00000000000000000000000000": "HLSCOPE",
}

INCEPTION_ANCHOR_FILE = "inception_anchor.json"

ORACLE_ID_HEX_SET = {oid[2:].lower() for oid in ORACLE_IDS.keys()}
ORACLE_ID_TO_FEED = {oid[2:].lower(): name for oid, name in ORACLE_IDS.items()}

NAV_DECIMALS = 8
REDSTONE_ADAPTER_CREATION_BLOCK = 20419584  # 20419584 = real Redstone creation; can also run 24035770 for STAC only

DEFAULT_BATCH_SIZE = 2000
MIN_BATCH_SIZE = 250
LLAMA_MAX_BLOCK_RANGE = 1000  # llamaRPC eth_getLogs hard range cap

TEST_MODE = "--test" in sys.argv
TEST_SAMPLE_SIZE = 10

ALLOW_SKIP = "--allow-skip" in sys.argv  # optional: skip bad ranges for speed over precision

INCREMENTAL_MODE = "--incremental" in sys.argv
LAST_BLOCK_FILE = "last_processed_block.txt"

OUTPUT_FILENAME = "nav_volatility_analysis_test.xlsx" if TEST_MODE else "nav_volatility_analysis.xlsx"

CONTRACT_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "uint256", "name": "value", "type": "uint256"},
            {"indexed": False, "internalType": "bytes32", "name": "dataFeedId", "type": "bytes32"},
            {"indexed": False, "internalType": "uint256", "name": "updatedAt", "type": "uint256"},
        ],
        "name": "ValueUpdate",
        "type": "event",
    }
]

EVENT_SIG_HASH = Web3.keccak(text="ValueUpdate(uint256,bytes32,uint256)")

# =============================
# RPC management
# =============================

_current_rpc_idx = 0
w3 = None

# rotate cooldown to avoid rapid RPC ping-pong
_last_rotate_ts = 0.0
ROTATE_COOLDOWN_SEC = 10


def connect_web3() -> Web3:
    global w3
    rpc = RPCS[_current_rpc_idx]
    candidate = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 30}))
    if not candidate.is_connected():
        raise Exception(f"Failed to connect to RPC: {rpc}")
    w3 = candidate
    return w3


def rotate_rpc(reason: str = "") -> None:
    """
    Rotate to the next *reachable* RPC.
    Tries all RPCs once before failing.
    """
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


def current_rpc() -> str:
    return RPCS[_current_rpc_idx]


def current_rpc_is_llama() -> bool:
    return "llamarpc" in current_rpc().lower()


def effective_batch_size(base_batch_size: int) -> int:
    if current_rpc_is_llama():
        return min(base_batch_size, LLAMA_MAX_BLOCK_RANGE)
    return base_batch_size


# =============================
# Helpers
# =============================

def load_inception_anchors(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # expected: {feed: {"date":"YYYY-MM-DD","nav":float}}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Warning: could not parse {path}: {e}")
        return {}


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


def safe_datafeed_hex(datafeed_id) -> str:
    if hasattr(datafeed_id, "hex"):
        h = datafeed_id.hex().lower()
    elif isinstance(datafeed_id, (bytes, bytearray)):
        h = datafeed_id.hex().lower()
    else:
        h = str(datafeed_id).lower()
    if h.startswith("0x"):
        h = h[2:]
    return h


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


def get_logs_resilient(params: dict, max_retries: int = 8):
    for attempt in range(max_retries):
        try:
            return w3.eth.get_logs(params)
        except Exception as e:
            msg = str(e)
            msg_l = msg.lower()

            if is_range_too_large_error(msg):
                raise

            # explicit auth issues
            if (
                "unauthorized" in msg_l
                or "authenticate your request" in msg_l
                or "invalid api key" in msg_l
                or "forbidden" in msg_l
            ):
                rotate_rpc("auth error on get_logs")
                wait = min(20, 2 ** attempt)
                print(f"  Auth error. Sleeping {wait}s; retrying same batch...")
                time.sleep(wait)
                continue

            # generic bad request -> rotate after first retry
            if "400 client error" in msg_l or "bad request" in msg_l:
                if attempt >= 1:
                    rotate_rpc("persistent 400 on get_logs")
                wait = min(20, 2 ** attempt)
                print(f"  400/BadRequest. Sleeping {wait}s; retrying same batch...")
                time.sleep(wait)
                continue

            if is_transient_error(msg):
                if is_rate_limit_error(msg):
                    rotate_rpc("rate limited on get_logs")
                wait = min(45, 2 ** attempt)
                print(f"  Transient error. Sleeping {wait}s; retrying same batch...")
                time.sleep(wait)
                continue

            raise

    raise Exception("Max retries exceeded in get_logs_resilient")


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
                wait = min(20, 2 ** attempt)
                time.sleep(wait)
                continue
            raise

    raise Exception(f"Failed to fetch timestamp for block {block_number}")


def load_existing_daily_nav(filename: str, feeds) -> dict:
    result = {feed: {} for feed in feeds}

    if not os.path.exists(filename):
        return result

    try:
        xls = pd.ExcelFile(filename, engine="openpyxl")
    except Exception:
        print(f"Warning: could not open existing file {filename}; continuing without merge.")
        return result

    for feed in feeds:
        if feed not in xls.sheet_names:
            continue
        try:
            df = pd.read_excel(xls, sheet_name=feed)
        except Exception:
            continue

        if "Date" not in df.columns or "NAV/Share" not in df.columns:
            continue

        dts = pd.to_datetime(df["Date"], errors="coerce")
        navs = pd.to_numeric(df["NAV/Share"], errors="coerce")

        for d, nav in zip(dts, navs):
            if pd.notna(d) and pd.notna(nav):
                result[feed][d.date()] = float(nav)

    return result


def build_metrics_dataframe(daily_nav: dict, feed_name: str, anchors: dict) -> pd.DataFrame:
    base_cols = [
        "Date", "NAV/Share", "Daily % change", "daily_return",
        "APY 7D", "APY 30D", "APY 90D", "APY Since Inception", "Sharpe Ratio"
    ]

    if not daily_nav:
        return pd.DataFrame(columns=base_cols)

    # Build base frame from daily NAV map
    df = pd.DataFrame(
        [{"Date": d, "NAV/Share": float(v)} for d, v in sorted(daily_nav.items())]
    )

    # Hard-trim to anchor date FIRST so all downstream metrics use anchored history
    feed_anchor = anchors.get(feed_name)
    if isinstance(feed_anchor, dict):
        try:
            a_date = pd.to_datetime(feed_anchor.get("date")).date()
            date_series = pd.to_datetime(df["Date"], errors="coerce").dt.date
            df = df[date_series >= a_date].copy()
            df = df.sort_values("Date").reset_index(drop=True)
        except Exception:
            pass

    if df.empty:
        return pd.DataFrame(columns=base_cols)

    # Recompute daily return AFTER trimming
    df["prev_nav"] = df["NAV/Share"].shift(1)
    df["daily_return"] = (df["NAV/Share"] / df["prev_nav"]) - 1.0
    df.loc[df["prev_nav"].isna() | (df["prev_nav"] == 0), "daily_return"] = pd.NA
    df["Daily % change"] = df["daily_return"] * 100.0
    df = df.drop(columns=["prev_nav"])

    # Rolling APYs
    for window in (7, 30, 90):
        col = f"APY {window}D"
        df[col] = (
            (1 + df["daily_return"])
            .rolling(window=window, min_periods=window)
            .apply(lambda x: (x.prod() ** (365.0 / window)) - 1.0, raw=False)
        ) * 100.0

    # Since inception APY (anchor-aware)
    inception_apy = pd.NA
    if len(df) >= 2:
        first_date = pd.to_datetime(df["Date"].iloc[0]).date()
        first_nav = float(df["NAV/Share"].iloc[0])

        # Prefer anchor NAV/date if present and valid
        if isinstance(feed_anchor, dict):
            try:
                a_date = pd.to_datetime(feed_anchor.get("date")).date()
                a_nav = float(feed_anchor.get("nav"))
                mask = pd.to_datetime(df["Date"], errors="coerce").dt.date == a_date
                if mask.any():
                    a_nav = float(df.loc[mask, "NAV/Share"].iloc[-1])
                if a_nav > 0:
                    first_date = a_date
                    first_nav = a_nav
            except Exception:
                pass

        last_date = pd.to_datetime(df["Date"].iloc[-1]).date()
        last_nav = float(df["NAV/Share"].iloc[-1])
        days_elapsed = (last_date - first_date).days

        if first_nav > 0 and last_nav > 0 and days_elapsed >= 7:
            inception_apy = ((last_nav / first_nav) ** (365.0 / days_elapsed) - 1.0) * 100.0

    df["APY Since Inception"] = inception_apy

    # Sharpe ratio (annualised, rf ~ 0)
    sharpe = pd.NA
    dr = pd.to_numeric(df["daily_return"], errors="coerce").dropna()
    if len(dr) >= 7:
        std_daily = dr.std()
        if pd.notna(std_daily) and std_daily > 0:
            sharpe = (dr.mean() / std_daily) * (365.0 ** 0.5)

    df["Sharpe Ratio"] = sharpe

    return df[base_cols]



def latest_non_null(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.iloc[-1] if len(s) > 0 else None


# =============================
# Main
# =============================

overall_start_ts = time.time()

connect_web3()
print(f"Connected via RPC: {current_rpc()}")

latest_block = w3.eth.block_number
print(f"Latest block: {latest_block}")

from_block = load_from_block(REDSTONE_ADAPTER_CREATION_BLOCK, latest_block)

if from_block > latest_block:
    print("Nothing new to fetch (from_block > latest_block).")
    sys.exit(0)

total_block_span = (latest_block - from_block + 1)

print(f"\nFetching ValueUpdate events for feeds: {', '.join(ORACLE_IDS.values())}")
print(f"Querying blocks {from_block} -> {latest_block} (span: {total_block_span:,} blocks)")

if TEST_MODE:
    print(f"TEST MODE: limiting final output to first {TEST_SAMPLE_SIZE} rows per vault")

anchors = load_inception_anchors(INCEPTION_ANCHOR_FILE)
if anchors:
    print(f"Loaded inception anchors from {INCEPTION_ANCHOR_FILE}: {', '.join(sorted(anchors.keys()))}")
else:
    print("No inception anchors loaded; using dataset-first-date per vault.")

contract = w3.eth.contract(
    address=Web3.to_checksum_address(REDSTONE_MULTIFEED_ADAPTER),
    abi=CONTRACT_ABI,
)
value_update_event = contract.events.ValueUpdate

batch_size = DEFAULT_BATCH_SIZE
current_from = from_block
total_logs = 0
nav_updates = []
block_timestamp_cache = {}

print("\nQuerying in batches...")

# for progress/ETA
processed_blocks = 0
batch_counter = 0

while current_from <= latest_block:
    active_batch = effective_batch_size(batch_size)
    current_to = min(current_from + active_batch - 1, latest_block)

    params = {
        "fromBlock": current_from,
        "toBlock": current_to,
        "address": Web3.to_checksum_address(REDSTONE_MULTIFEED_ADAPTER),
        "topics": [EVENT_SIG_HASH],
    }

    try:
        logs = get_logs_resilient(params)
    except Exception as e:
        msg = str(e)

        if is_range_too_large_error(msg):
            new_batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
            if new_batch_size == batch_size:
                raise Exception(
                    f"Cannot reduce batch size further (already {batch_size}) but still hitting limit."
                ) from e

            print(
                f"  Range/result limit in {current_from}-{current_to}; "
                f"batch {batch_size} -> {new_batch_size}, retrying same range..."
            )
            batch_size = new_batch_size
            continue

        print(f"  Error querying blocks {current_from}-{current_to}: {e}")

        if ALLOW_SKIP:
            print("  --allow-skip enabled: skipping this range.")
            processed_blocks += (current_to - current_from + 1)
            current_from = current_to + 1
            continue

        print("  Strict mode: keeping same range, sleeping 15s, then retrying...")
        time.sleep(15)
        continue

    total_logs += len(logs)
    batch_counter += 1

    print(f"  Blocks {current_from}-{current_to}: Found {len(logs)} ValueUpdate events")

    for log in logs:
        try:
            decoded = value_update_event.process_log(log)
        except Exception:
            continue

        event_id_hex = safe_datafeed_hex(decoded.args.dataFeedId)
        if event_id_hex not in ORACLE_ID_HEX_SET:
            continue

        feed_name = ORACLE_ID_TO_FEED[event_id_hex]
        bn = decoded.blockNumber

        try:
            ts = get_block_timestamp_cached(bn, block_timestamp_cache)
            dt = datetime.fromtimestamp(ts)
        except Exception:
            dt = None

        nav_raw = decoded.args.value
        nav_per_share = nav_raw / (10 ** NAV_DECIMALS)

        nav_updates.append({
            "feed": feed_name,
            "datetime": dt,
            "nav_per_share": nav_per_share,
            "block_number": bn,
        })

    # Progress + ETA
    step_blocks = (current_to - current_from + 1)
    processed_blocks += step_blocks
    elapsed = time.time() - overall_start_ts
    progress = processed_blocks / total_block_span if total_block_span > 0 else 1.0
    eta = (elapsed / progress) - elapsed if progress > 0 else None
    expected_total = (elapsed / progress) if progress > 0 else None

    if batch_counter % 5 == 0 or current_to == latest_block:
        print(
            f"    Progress: {progress*100:.2f}% | "
            f"Elapsed: {fmt_seconds(elapsed)} | "
            f"ETA: {fmt_seconds(eta)} | "
            f"Expected total: {fmt_seconds(expected_total)} | "
            f"RPC: {current_rpc()}"
        )

    current_from = current_to + 1

print(f"\nTotal logs found: {total_logs}")
print(f"Matching NAV updates: {len(nav_updates)}")

# Build new daily NAV map
new_daily_nav = {feed: {} for feed in ORACLE_IDS.values()}

if nav_updates:
    nav_updates.sort(key=lambda x: x["block_number"])

for upd in nav_updates:
    if upd["datetime"] is None:
        continue
    d = upd["datetime"].date()
    new_daily_nav[upd["feed"]][d] = float(upd["nav_per_share"])  # last event/day wins

# Merge with existing history in incremental mode
if INCREMENTAL_MODE and not TEST_MODE:
    existing_daily_nav = load_existing_daily_nav(OUTPUT_FILENAME, ORACLE_IDS.values())
else:
    existing_daily_nav = {feed: {} for feed in ORACLE_IDS.values()}

merged_daily_nav = {}
for feed in ORACLE_IDS.values():
    combined = dict(existing_daily_nav.get(feed, {}))
    combined.update(new_daily_nav.get(feed, {}))
    merged_daily_nav[feed] = combined

# Fallback: seed from anchors when no merged history exists
if all(len(v) == 0 for v in merged_daily_nav.values()):
    anchors = load_inception_anchors(INCEPTION_ANCHOR_FILE)
    seeded_any = False
    for feed, cfg in anchors.items():
        if feed in merged_daily_nav:
            try:
                d = pd.to_datetime(cfg["date"]).date()
                nav = float(cfg["nav"])
                merged_daily_nav[feed][d] = nav
                seeded_any = True
            except Exception:
                pass

    if seeded_any:
        print("No fresh updates; seeded dataset from inception anchors.")
    else:
        print("\nNo NAV updates found and no existing history to merge.")
        print("You may need to adjust from_block or run once without --incremental.")
        sys.exit(0)


vault_frames = {}
for feed in ORACLE_IDS.values():
    df = build_metrics_dataframe(merged_daily_nav[feed], feed, anchors)
    if TEST_MODE and not df.empty:
        df = df.head(TEST_SAMPLE_SIZE).copy()
    vault_frames[feed] = df
    print(f"  {feed}: {len(df)} daily rows")

print(f"\nExporting to {OUTPUT_FILENAME}...")

with pd.ExcelWriter(OUTPUT_FILENAME, engine="openpyxl") as writer:
    for feed in sorted(vault_frames.keys()):
        df = vault_frames[feed].copy()

        output_cols = [
            "Date",
            "NAV/Share",
            "Daily % change",
            "APY 7D",
            "APY 30D",
            "APY 90D",
            "APY Since Inception",
            "Sharpe Ratio",
        ]

        if df.empty:
            pd.DataFrame(columns=output_cols).to_excel(writer, sheet_name=feed, index=False)
            continue

        for col in output_cols:
            if col not in df.columns:
                df[col] = pd.NA

        df = df[output_cols]

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%m/%d/%Y")
        df["NAV/Share"] = pd.to_numeric(df["NAV/Share"], errors="coerce").round(6)

        pct_cols = ["Daily % change", "APY 7D", "APY 30D", "APY 90D", "APY Since Inception"]
        for col in pct_cols:
            df[col] = df[col].apply(lambda x: f"{x:.5f}%" if pd.notna(x) else "")

        df["Sharpe Ratio"] = df["Sharpe Ratio"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

        df.to_excel(writer, sheet_name=feed, index=False)
        print(f"  Created sheet '{feed}' with {len(df)} rows")

print(f"\nSuccessfully exported: {OUTPUT_FILENAME}")

if INCREMENTAL_MODE:
    save_last_block(latest_block)
    print(f"Saved last processed block: {latest_block} -> {LAST_BLOCK_FILE}")

print("\n" + "=" * 110)
print("SUMMARY")
print("=" * 110)

for feed in sorted(vault_frames.keys()):
    raw_df = vault_frames[feed]
    print(f"\n{feed}:")
    print(f"  Total days: {len(raw_df)}")

    if raw_df.empty:
        print("  No data")
        continue

    changes = pd.to_numeric(raw_df["Daily % change"], errors="coerce").dropna()
    navs = pd.to_numeric(raw_df["NAV/Share"], errors="coerce").dropna()

    if len(changes) > 0:
        print(f"  Max Loss: {changes.min():.5f}%")
        print(f"  Daily Std Dev: {changes.std():.8f}")
    else:
        print("  Max Loss: n/a")
        print("  Daily Std Dev: n/a")

    if len(navs) > 0:
        print(f"  NAV Range: {navs.min():.6f} - {navs.max():.6f}")

    # Backwards-compatible inception column lookup (old/new naming)
    apy_inc_col = None
    if "APY Since Inception" in raw_df.columns:
        apy_inc_col = "APY Since Inception"
    elif "APY Since Inception (dataset)" in raw_df.columns:
        apy_inc_col = "APY Since Inception (dataset)"

    apy7 = latest_non_null(raw_df["APY 7D"]) if "APY 7D" in raw_df.columns else None
    apy30 = latest_non_null(raw_df["APY 30D"]) if "APY 30D" in raw_df.columns else None
    apy90 = latest_non_null(raw_df["APY 90D"]) if "APY 90D" in raw_df.columns else None
    apy_inc = latest_non_null(raw_df[apy_inc_col]) if apy_inc_col else None
    sharpe = latest_non_null(raw_df["Sharpe Ratio"]) if "Sharpe Ratio" in raw_df.columns else None

    print(f"  Latest APY 7D: {apy7:.3f}%" if apy7 is not None else "  Latest APY 7D: n/a")
    print(f"  Latest APY 30D: {apy30:.3f}%" if apy30 is not None else "  Latest APY 30D: n/a")
    print(f"  Latest APY 90D: {apy90:.3f}%" if apy90 is not None else "  Latest APY 90D: n/a")
    print(f"  APY Since Inception: {apy_inc:.3f}%" if apy_inc is not None else "  APY Since Inception: n/a")
    print(f"  Sharpe Ratio: {sharpe:.4f}" if sharpe is not None else "  Sharpe Ratio: n/a")


total_elapsed = time.time() - overall_start_ts
print(f"\nDone. Total runtime: {fmt_seconds(total_elapsed)}")
