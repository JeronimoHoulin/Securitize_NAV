
import os
import sys
from datetime import datetime, date
from collections import defaultdict
from dotenv import load_dotenv
from web3 import Web3
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment variables
api_key = os.getenv('api_key')
infura_base_url = 'https://mainnet.infura.io/v3/'

#Oracle params
redstone_multifeed_adapter = '0xd72a6ba4a87ddb33e801b3f1c7750b2d0911fc6c'

# Define all oracle IDs and their corresponding feed names
ORACLE_IDS = {
    '0x535441435f46554e44414d454e54414c00000000000000000000000000000000': 'STAC',
    '0x41435245445f46554e44414d454e54414c000000000000000000000000000000': 'ACRED',
    '0x484c53636f70655f46554e44414d454e54414c00000000000000000000000000': 'HLSCOPE'
}

# Create a set of oracle_id hex strings (lowercase) for fast lookup
oracle_id_hex_set = {oracle_id[2:].lower() for oracle_id in ORACLE_IDS.keys()}

# Create a reverse lookup: hex string (no 0x) -> feed name
oracle_id_to_feed = {oracle_id[2:].lower(): feed for oracle_id, feed in ORACLE_IDS.items()}

# Configuration
# NAV values are stored with 8 decimal places (common for financial values)
NAV_DECIMALS = 8
TEST_MODE = '--test' in sys.argv
TEST_SAMPLE_SIZE = 10  # Number of samples per vault for test mode

# Redstone multifeed adapter was created at block 20419584
# We only query from this block onwards since NAV updates only exist after the adapter was deployed
REDSTONE_ADAPTER_CREATION_BLOCK = 20419584


# Connect to Infura
infura_url = f"{infura_base_url}{api_key}"
w3 = Web3(Web3.HTTPProvider(infura_url))

# Verify connection
if not w3.is_connected():
    raise Exception("Failed to connect to Infura")

print(f"Connected to Ethereum mainnet via Infura")
print(f"Latest block: {w3.eth.block_number}")

# Contract ABI for Redstone multifeed adapter implementation
# Using the implementation ABI since it's a proxy contract
# The ValueUpdate event has dataFeedId as non-indexed, so we'll filter after fetching
CONTRACT_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "uint256", "name": "value", "type": "uint256"},
            {"indexed": False, "internalType": "bytes32", "name": "dataFeedId", "type": "bytes32"},
            {"indexed": False, "internalType": "uint256", "name": "updatedAt", "type": "uint256"}
        ],
        "name": "ValueUpdate",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "bytes32", "name": "dataFeedId", "type": "bytes32"}
        ],
        "name": "UpdateSkipDueToBlockTimestamp",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "bytes32", "name": "dataFeedId", "type": "bytes32"}
        ],
        "name": "UpdateSkipDueToDataTimestamp",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "bytes32", "name": "dataFeedId", "type": "bytes32"}
        ],
        "name": "UpdateSkipDueToInvalidValue",
        "type": "event"
    }
]

# Create contract instance
contract = w3.eth.contract(address=Web3.to_checksum_address(redstone_multifeed_adapter), abi=CONTRACT_ABI)

# Get all ValueUpdate events (dataFeedId is not indexed, so we filter after fetching)
print(f"\nFetching ValueUpdate events for feeds: {', '.join(ORACLE_IDS.values())}")
print(f"Querying from block {REDSTONE_ADAPTER_CREATION_BLOCK} (Redstone multifeed adapter creation)...")

# Get events - query from block 20419584 (when Redstone multifeed adapter was created)
# NAV updates only exist after the adapter was deployed
latest_block = w3.eth.block_number
from_block = REDSTONE_ADAPTER_CREATION_BLOCK
print(f"Querying from block {from_block} (Redstone multifeed adapter creation) to {latest_block}...")
to_block = 'latest'

if TEST_MODE:
    print(f"TEST MODE: Will limit output to first {TEST_SAMPLE_SIZE} samples per vault")

# Query all ValueUpdate events in batches to avoid 10,000 result limit
nav_updates = []
try:
    # Get the event signature hash for ValueUpdate event
    # Event signature: ValueUpdate(uint256,bytes32,uint256)
    value_update_event = contract.events.ValueUpdate
    event_signature = "ValueUpdate(uint256,bytes32,uint256)"
    event_signature_hash = Web3.keccak(text=event_signature)
    
    # Query in smaller batches to avoid hitting the 10,000 result limit
    batch_size = 5000  # Query 5000 blocks at a time
    current_from = from_block
    total_logs = 0
    
    print("Querying in batches to avoid result limit...")
    
    while current_from <= latest_block:
        current_to = min(current_from + batch_size - 1, latest_block)
        
        try:
            # Use w3.eth.get_logs() with the event filter
            logs = w3.eth.get_logs({
                'fromBlock': current_from,
                'toBlock': current_to,
                'address': Web3.to_checksum_address(redstone_multifeed_adapter),
                'topics': [event_signature_hash]  # Filter by event signature
            })
            
            total_logs += len(logs)
            print(f"  Blocks {current_from}-{current_to}: Found {len(logs)} ValueUpdate events")
            
            # Decode and filter events where dataFeedId matches any of our oracle IDs
            for log in logs:
                try:
                    # Decode the log using the contract event
                    decoded_event = value_update_event.process_log(log)
                    event_data_feed_id = decoded_event.args.dataFeedId
                    
                    # Convert to hex string for comparison (handles HexBytes, bytes, etc.)
                    if hasattr(event_data_feed_id, 'hex'):
                        event_id_hex = event_data_feed_id.hex().lower()
                    elif isinstance(event_data_feed_id, bytes):
                        event_id_hex = event_data_feed_id.hex().lower()
                    else:
                        event_id_hex = str(event_data_feed_id).lower()
                    
                    # Remove '0x' prefix if present for comparison
                    if event_id_hex.startswith('0x'):
                        event_id_hex = event_id_hex[2:]
                    
                    # Check if this event matches any of our oracle IDs
                    if event_id_hex in oracle_id_hex_set:
                        # Get the feed name for this oracle ID
                        feed_name = oracle_id_to_feed.get(event_id_hex)
                        
                        # Get block timestamp for date grouping
                        try:
                            block = w3.eth.get_block(decoded_event.blockNumber)
                            block_timestamp = block['timestamp']
                            dt = datetime.fromtimestamp(block_timestamp)
                        except:
                            # If we can't get timestamp now, we'll get it later
                            dt = None
                        
                        # Convert NAV value to NAV/Share (divide by 10^8 for 8 decimals)
                        nav_value_raw = decoded_event.args.value
                        nav_per_share = nav_value_raw / (10 ** NAV_DECIMALS)
                        
                        # Store the event with feed name, date, and NAV/Share
                        nav_updates.append({
                            'event': decoded_event,
                            'feed': feed_name,
                            'datetime': dt,
                            'nav_per_share': nav_per_share,
                            'block_number': decoded_event.blockNumber
                        })
                except Exception as decode_error:
                    # Skip logs that can't be decoded (shouldn't happen, but just in case)
                    continue
            
            # Move to next batch
            current_from = current_to + 1
            
        except Exception as batch_error:
            # If we hit the limit, try smaller batches
            error_msg = str(batch_error)
            if 'more than 10000 results' in error_msg or '10000' in error_msg:
                print(f"  Hit result limit for blocks {current_from}-{current_to}, trying smaller batch...")
                # Try with half the batch size
                batch_size = max(1000, batch_size // 2)
                continue
            else:
                # Other error, try to continue with next batch
                print(f"  Error querying blocks {current_from}-{current_to}: {batch_error}")
                current_from = current_to + 1
                continue
    
    print(f"\nTotal: Found {total_logs} ValueUpdate event logs across all batches")
    print(f"Found {len(nav_updates)} NAV updates matching any of the oracle IDs")
    
except Exception as e:
    print(f"Error querying ValueUpdate events: {e}")
    import traceback
    traceback.print_exc()

# Sort by block number (oldest first)
if nav_updates:
    nav_updates.sort(key=lambda x: x['block_number'])
    
    # Get timestamps for any updates that don't have them yet
    print("\nFetching block timestamps for NAV updates...")
    for update in nav_updates:
        if update['datetime'] is None:
            try:
                block = w3.eth.get_block(update['block_number'])
                block_timestamp = block['timestamp']
                update['datetime'] = datetime.fromtimestamp(block_timestamp)
            except:
                print(f"Warning: Could not get timestamp for block {update['block_number']}")
    
    # Group NAV updates by feed and date
    # For each date, we'll use the last NAV value of that day (to handle intra-day changes)
    print("\nProcessing NAV updates and calculating daily changes...")
    
    vault_data = defaultdict(list)  # feed -> list of {date, nav_per_share, daily_pct_change}
    
    for feed_name in ORACLE_IDS.values():
        # Get all updates for this feed
        feed_updates = [u for u in nav_updates if u['feed'] == feed_name]
        
        if not feed_updates:
            print(f"No updates found for {feed_name}")
            continue
        
        # Group by date (use the last NAV value of each day)
        daily_nav = {}  # date -> nav_per_share (last value of the day)
        
        for update in feed_updates:
            if update['datetime'] is None:
                continue
            
            update_date = update['datetime'].date()
            # Keep the last NAV value for each date (since we're sorted by block number)
            daily_nav[update_date] = update['nav_per_share']
        
        # Sort dates
        sorted_dates = sorted(daily_nav.keys())
        
        # Calculate daily % changes
        previous_nav = None
        for current_date in sorted_dates:
            current_nav = daily_nav[current_date]
            
            if previous_nav is not None:
                # Calculate daily % change
                daily_pct_change = ((current_nav - previous_nav) / previous_nav) * 100
            else:
                # First day has no previous value
                daily_pct_change = None
            
            vault_data[feed_name].append({
                'Date': current_date,
                'NAV/Share': current_nav,
                'Daily % change': daily_pct_change
            })
            
            previous_nav = current_nav
        
        print(f"  {feed_name}: {len(vault_data[feed_name])} daily records")
    
    # Apply test mode limit if enabled
    if TEST_MODE:
        print(f"\nApplying test mode: limiting to first {TEST_SAMPLE_SIZE} samples per vault...")
        for feed_name in vault_data:
            vault_data[feed_name] = vault_data[feed_name][:TEST_SAMPLE_SIZE]
    
    # Create Excel file with one sheet per vault (CSV doesn't support multiple tabs)
    output_filename = 'nav_volatility_analysis_test.xlsx' if TEST_MODE else 'nav_volatility_analysis.xlsx'
    print(f"\nExporting to {output_filename}...")
    
    # Create Excel writer for multiple sheets
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        for feed_name in sorted(vault_data.keys()):
            df = pd.DataFrame(vault_data[feed_name])
            
            # Format the data
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%m/%d/%Y')
            df['NAV/Share'] = df['NAV/Share'].round(6)
            df['Daily % change'] = df['Daily % change'].apply(
                lambda x: f"{x:.5f}%" if x is not None else ""
            )
            
            # Write to sheet
            df.to_excel(writer, sheet_name=feed_name, index=False)
            print(f"  Created sheet '{feed_name}' with {len(df)} rows")
    
    print(f"\nSuccessfully exported volatility analysis to {output_filename}")
    print(f"  Total vaults: {len(vault_data)}")
    
    # Also print a summary to console
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}\n")
    
    for feed_name in sorted(vault_data.keys()):
        data = vault_data[feed_name]
        print(f"{feed_name}:")
        print(f"  Total days: {len(data)}")
        if data:
            changes = [d['Daily % change'] for d in data if d['Daily % change'] is not None]
            if changes:
                max_loss = min(changes)
                daily_std = pd.Series(changes).std()
                print(f"  Max Loss: {max_loss:.5f}%")
                print(f"  Daily Std Dev: {daily_std:.8f}")
            print(f"  NAV Range: {min(d['NAV/Share'] for d in data):.6f} - {max(d['NAV/Share'] for d in data):.6f}")
        print()
    
else:
    print(f"\nNo NAV updates found for any of the oracle IDs")
    print("You may need to adjust the from_block range or check if the contract has emitted events.")

