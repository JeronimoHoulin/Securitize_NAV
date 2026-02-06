import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from web3 import Web3
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment variables
api_key = os.getenv('api_key')
infura_base_url = 'https://mainnet.infura.io/v3/'

# Contract addresses
VBILL_TOKEN_ADDRESS = '0x2255718832bc9fd3be1caf75084f4803da14ff01'
ISSUER_CONTRACT_ADDRESS = '0x22afdb66dc56be3a81285d953124bda8020dcb88'
NULL_ADDRESS = '0x0000000000000000000000000000000000000000'

# Contract creation block
ISSUER_CONTRACT_CREATION_BLOCK = 22468524

# Function method IDs
BULK_ISSUANCE = '0xb28d07c3'  # This is interest

# Configuration
TEST_MODE = '--test' in sys.argv

# Connect to Infura
infura_url = f"{infura_base_url}{api_key}"
w3 = Web3(Web3.HTTPProvider(infura_url))

# Verify connection
if not w3.is_connected():
    raise Exception("Failed to connect to Infura")

print(f"Connected to Ethereum mainnet via Infura")
print(f"Latest block: {w3.eth.block_number}")

# ERC-20 Transfer event ABI
TRANSFER_EVENT_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True, "internalType": "address", "name": "from", "type": "address"},
        {"indexed": True, "internalType": "address", "name": "to", "type": "address"},
        {"indexed": False, "internalType": "uint256", "name": "value", "type": "uint256"}
    ],
    "name": "Transfer",
    "type": "event"
}

# ERC-20 standard ABI for decimals() function
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    }
]

# Create VBILL token contract instance
vbill_token_full = w3.eth.contract(address=Web3.to_checksum_address(VBILL_TOKEN_ADDRESS), abi=ERC20_ABI + [TRANSFER_EVENT_ABI])
vbill_token = w3.eth.contract(address=Web3.to_checksum_address(VBILL_TOKEN_ADDRESS), abi=[TRANSFER_EVENT_ABI])

# Get token decimals
try:
    VBILL_DECIMALS = vbill_token_full.functions.decimals().call()
    print(f"VBILL token decimals: {VBILL_DECIMALS}")
except Exception as e:
    print(f"Warning: Could not query VBILL decimals, defaulting to 18. Error: {e}")
    VBILL_DECIMALS = 18  # Default to 18 if we can't query

# Transfer event signature hash
transfer_event_signature = Web3.keccak(text="Transfer(address,address,uint256)")

# Null address topic (padded to 32 bytes)
null_address_topic = Web3.to_hex(Web3.to_bytes(hexstr=NULL_ADDRESS).rjust(32, b'\x00'))

print(f"\nStep 1: Getting all VBILL mint events (from null address)")
print(f"Querying from block {ISSUER_CONTRACT_CREATION_BLOCK}...")

from_block = ISSUER_CONTRACT_CREATION_BLOCK
if TEST_MODE:
    # Test mode: query approximately 1 month of blocks (~216,000 blocks)
    # Ethereum blocks are produced every ~12 seconds, so ~7,200 blocks per day
    # 1 month = 30 days * 7,200 blocks/day = 216,000 blocks
    to_block = from_block + 216000
    print(f"TEST MODE: Querying blocks {from_block} to {to_block} (~1 month, {to_block - from_block} blocks)")
else:
    to_block = w3.eth.block_number
    print(f"Querying blocks {from_block} to {to_block}...")

print("\nFetching mint events in batches...")

batch_size = 10000  # Query 10000 blocks at a time
current_from = from_block
mint_events = []  # List of all mint events

while current_from <= to_block:
    current_to = min(current_from + batch_size - 1, to_block)
    
    try:
        # Query Transfer events where from=null (minting)
        logs_mint = w3.eth.get_logs({
            'fromBlock': current_from,
            'toBlock': current_to,
            'address': Web3.to_checksum_address(VBILL_TOKEN_ADDRESS),
            'topics': [transfer_event_signature, null_address_topic]  # from=null
        })
        
        for log in logs_mint:
            try:
                decoded = vbill_token.events.Transfer().process_log(log)
                tx_hash = log['transactionHash'].hex()
                
                mint_events.append({
                    'log': log,
                    'from': decoded['args']['from'],
                    'to': decoded['args']['to'],
                    'value': decoded['args']['value'],
                    'block_number': log['blockNumber'],
                    'tx_hash': tx_hash
                })
            except Exception as e:
                continue
        
        print(f"  Blocks {current_from}-{current_to}: Found {len(logs_mint)} mint events (total: {len(mint_events)})")
        current_from = current_to + 1
        
    except Exception as batch_error:
        error_msg = str(batch_error)
        if 'more than 10000 results' in error_msg or '10000' in error_msg:
            print(f"  Hit result limit for blocks {current_from}-{current_to}, trying smaller batch...")
            batch_size = max(1000, batch_size // 2)
            continue
        else:
            print(f"  Error querying blocks {current_from}-{current_to}: {batch_error}")
            current_from = current_to + 1
            continue

print(f"\nTotal mint events found: {len(mint_events)}")

# Step 2: Classify each mint event
print(f"\nStep 2: Classifying mint events by transaction type...")

for i, event in enumerate(mint_events):
    try:
        tx = w3.eth.get_transaction(event['tx_hash'])
        
        # Check if transaction calls bulkIssuance on issuer contract
        is_interest = False
        if tx['to'] and tx['to'].lower() == Web3.to_checksum_address(ISSUER_CONTRACT_ADDRESS).lower():
            if tx['input']:
                # Handle both hex string and bytes
                if isinstance(tx['input'], bytes):
                    if len(tx['input']) >= 4:
                        method_id_hex = tx['input'][:4].hex()
                    else:
                        continue
                else:
                    # Hex string
                    if len(tx['input']) >= 10:
                        method_id_hex = tx['input'][:10]
                    else:
                        continue
                
                # Normalize to lowercase hex string without 0x prefix for comparison
                if method_id_hex.startswith('0x'):
                    method_id = method_id_hex[2:].lower()
                else:
                    method_id = method_id_hex.lower()
                
                # Compare with BULK_ISSUANCE (without 0x prefix)
                bulk_issuance_id = BULK_ISSUANCE[2:].lower() if BULK_ISSUANCE.startswith('0x') else BULK_ISSUANCE.lower()
                if method_id == bulk_issuance_id:
                    is_interest = True
        
        event['is_interest'] = is_interest
        
        if (i + 1) % 100 == 0:
            print(f"  Classified {i + 1}/{len(mint_events)} events...")
            
    except Exception as e:
        event['is_interest'] = False
        continue

# Count interest vs non-interest
interest_count = sum(1 for e in mint_events if e.get('is_interest', False))
print(f"  Classified {interest_count} as interest (bulkIssuance), {len(mint_events) - interest_count} as supply increase")

# Step 3: Get burn events (to null address)
print(f"\nStep 3: Getting burn events (to null address)...")

current_from = from_block
burn_events = []

while current_from <= to_block:
    current_to = min(current_from + batch_size - 1, to_block)
    
    try:
        # Query Transfer events where to=null (burning)
        logs_burn = w3.eth.get_logs({
            'fromBlock': current_from,
            'toBlock': current_to,
            'address': Web3.to_checksum_address(VBILL_TOKEN_ADDRESS),
            'topics': [transfer_event_signature, None, null_address_topic]  # to=null
        })
        
        for log in logs_burn:
            try:
                decoded = vbill_token.events.Transfer().process_log(log)
                burn_events.append({
                    'log': log,
                    'from': decoded['args']['from'],
                    'to': decoded['args']['to'],
                    'value': decoded['args']['value'],
                    'block_number': log['blockNumber'],
                    'tx_hash': log['transactionHash'].hex()
                })
            except Exception as e:
                continue
        
        print(f"  Blocks {current_from}-{current_to}: Found {len(logs_burn)} burn events (total: {len(burn_events)})")
        current_from = current_to + 1
        
    except Exception as batch_error:
        error_msg = str(batch_error)
        if 'more than 10000 results' in error_msg or '10000' in error_msg:
            print(f"  Hit result limit for blocks {current_from}-{current_to}, trying smaller batch...")
            batch_size = max(1000, batch_size // 2)
            continue
        else:
            print(f"  Error querying blocks {current_from}-{current_to}: {batch_error}")
            current_from = current_to + 1
            continue

print(f"\nTotal burn events found: {len(burn_events)}")

# Step 4: Group by date and calculate daily aggregates
print(f"\nStep 4: Grouping by date and calculating daily supply changes...")

# Combine all events and sort by block number
all_events = []
for event in mint_events:
    all_events.append({
        'type': 'mint',
        'is_interest': event.get('is_interest', False),
        'value': event['value'],
        'block_number': event['block_number'],
        'tx_hash': event['tx_hash']
    })

for event in burn_events:
    all_events.append({
        'type': 'burn',
        'is_interest': False,  # Burns are never interest
        'value': event['value'],
        'block_number': event['block_number'],
        'tx_hash': event['tx_hash']
    })

all_events.sort(key=lambda x: x['block_number'])

# Group by date
daily_data = {}
current_supply = 0  # Track current supply (in wei)

for event in all_events:
    try:
        block_number = event['block_number']
        block = w3.eth.get_block(block_number)
        block_timestamp = block['timestamp']
        dt = datetime.fromtimestamp(block_timestamp)
        date = dt.date()
        
        # Initialize daily data if needed
        if date not in daily_data:
            daily_data[date] = {
                'interest_minted': 0,  # Interest from bulkIssuance
                'supply_minted': 0,    # Other mints (supply increase)
                'burned': 0,            # All burns (supply decrease)
                'blocks': [],
                'timestamps': []
            }
        
        if event['type'] == 'mint':
            if event['is_interest']:
                daily_data[date]['interest_minted'] += event['value']
            else:
                daily_data[date]['supply_minted'] += event['value']
        elif event['type'] == 'burn':
            daily_data[date]['burned'] += event['value']
        
        daily_data[date]['blocks'].append(block_number)
        daily_data[date]['timestamps'].append(dt)
        
    except Exception as e:
        continue

# Calculate daily NAV and yield
print(f"\nCalculating daily NAV and yield...")

nav_data = []
sorted_dates = sorted(daily_data.keys())

for date in sorted_dates:
    data = daily_data[date]
    
    interest_minted = data['interest_minted']
    supply_minted = data['supply_minted']
    burned = data['burned']
    
    # Total supply change
    net_supply_change = interest_minted + supply_minted - burned
    
    # Update current supply
    previous_supply = current_supply
    current_supply += net_supply_change
    
    # Calculate daily yield (only from interest mints)
    if previous_supply > 0:
        daily_yield_pct = (interest_minted / previous_supply) * 100 if interest_minted > 0 else 0
    else:
        daily_yield_pct = None
    
    # Use the latest timestamp for the day
    latest_dt = max(data['timestamps'])
    
    nav_data.append({
        'Date': date,
        'DateTime': latest_dt,
        'Block': max(data['blocks']),
        'Interest_Minted': interest_minted / (10 ** VBILL_DECIMALS),  # Interest from bulkIssuance
        'Supply_Minted': supply_minted / (10 ** VBILL_DECIMALS),      # Other mints
        'Burned': burned / (10 ** VBILL_DECIMALS),                     # All burns
        'Net_Supply_Change': net_supply_change / (10 ** VBILL_DECIMALS),
        'Supply_Before': previous_supply / (10 ** VBILL_DECIMALS),
        'Supply_After': current_supply / (10 ** VBILL_DECIMALS),
        'Daily_Yield_%': daily_yield_pct,
        'NAV': 1.0  # NAV is always $1 for rebasing tokens
    })

print(f"\nProcessed {len(nav_data)} days of data")

# Convert to final output format
print(f"\nPreparing final output...")

final_data = []
for entry in nav_data:
    daily_pct_change = entry['Daily_Yield_%']
    
    final_data.append({
        'Date': entry['Date'].strftime('%m/%d/%Y'),
        'NAV/Share': 1.0,  # NAV is always $1 for rebasing tokens
        'Daily % change': f"{daily_pct_change:.6f}%" if daily_pct_change is not None else "",
        'Interest_Minted': round(entry['Interest_Minted'], 2),  # Interest from bulkIssuance
        'Supply_Minted': round(entry['Supply_Minted'], 2),        # Other mints
        'Burned': round(entry['Burned'], 2),                      # All burns
        'Net_Supply_Change': round(entry['Net_Supply_Change'], 2),
        'Supply': round(entry['Supply_After'], 2)
    })

# Create Excel file with VBILL sheet
output_filename = 'nav_volatility_analysis_test.xlsx' if TEST_MODE else 'nav_volatility_analysis.xlsx'
print(f"\nExporting to {output_filename}...")

# Check if file exists and read existing sheets
existing_sheets = {}
try:
    if os.path.exists(output_filename):
        existing_df = pd.read_excel(output_filename, sheet_name=None, engine='openpyxl')
        existing_sheets = existing_df
        print(f"  Found existing file with {len(existing_sheets)} sheets")
except Exception as e:
    print(f"  No existing file or error reading: {e}")

# Create Excel writer
with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    # Write VBILL sheet
    df_vbill = pd.DataFrame(final_data)
    df_vbill.to_excel(writer, sheet_name='VBILL', index=False)
    print(f"  Created/Updated sheet 'VBILL' with {len(df_vbill)} rows")
    
    # Write other existing sheets (if any)
    for sheet_name, df in existing_sheets.items():
        if sheet_name != 'VBILL':  # Don't overwrite, we already wrote VBILL
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Preserved sheet '{sheet_name}' with {len(df)} rows")

print(f"\nSuccessfully exported VBILL NAV analysis to {output_filename}")

# Print summary
print(f"\n{'='*100}")
print(f"VBILL NAV TRACKING SUMMARY")
print(f"{'='*100}\n")
print(f"Total days tracked: {len(final_data)}")
print(f"Total mint events: {len(mint_events)}")
print(f"Total burn events: {len(burn_events)}")
if final_data:
    interest_minted = sum(d['Interest_Minted'] for d in final_data)
    supply_minted = sum(d['Supply_Minted'] for d in final_data)
    burned = sum(d['Burned'] for d in final_data)
    final_supply = final_data[-1]['Supply']
    print(f"Total Interest Minted (bulkIssuance): {interest_minted:,.2f}")
    print(f"Total Supply Minted (other): {supply_minted:,.2f}")
    print(f"Total Burned: {burned:,.2f}")
    print(f"Final Supply: {final_supply:,.2f}")
    
    changes = [d['Daily % change'] for d in final_data if d['Daily % change']]
    if changes:
        # Parse percentage strings
        change_values = [float(c.replace('%', '')) for c in changes if c]
        if change_values:
            avg_yield = sum(change_values) / len(change_values)
            print(f"Average Daily Yield: {avg_yield:.6f}%")
            print(f"Max Daily Yield: {max(change_values):.6f}%")
            print(f"Min Daily Yield: {min(change_values):.6f}%")
print()
