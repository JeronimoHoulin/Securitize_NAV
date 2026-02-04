
import os
from datetime import datetime
from dotenv import load_dotenv
from web3 import Web3

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment variables
api_key = os.getenv('api_key')
infura_base_url = 'https://mainnet.infura.io/v3/'

#Oracle params
redstone_multifeed_adapter = '0xd72a6ba4a87ddb33e801b3f1c7750b2d0911fc6c'

vbill_id = '0x5642494c4c5f455448455245554d5f46554e44414d454e54414c000000000000' #VBILL
stac_id = '0x535441435f46554e44414d454e54414c00000000000000000000000000000000' #STAC
acred_id = '0x41435245445f46554e44414d454e54414c000000000000000000000000000000' #ACRED
hlscope_id = '0x484c53636f70655f46554e44414d454e54414c00000000000000000000000000' #HLSCOPE


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

# Convert oracle_id to bytes32 format for filtering
oracle_id_bytes = bytes.fromhex(oracle_id[2:])  # Remove '0x' prefix

# Get all ValueUpdate events (dataFeedId is not indexed, so we filter after fetching)
print(f"\nFetching ValueUpdate events for dataFeedId: {oracle_id}")
print("This may take a while if querying from genesis block...")

# Get events - query from a reasonable starting point (e.g., last 100k blocks)
# Adjust this based on when the contract was deployed
latest_block = w3.eth.block_number
from_block = max(0, latest_block - 100000)  # Last 100k blocks as starting point
to_block = 'latest'

print(f"Querying blocks {from_block} to {latest_block}...")

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
    
    oracle_id_hex = oracle_id[2:].lower()  # Remove '0x' and lowercase for comparison
    
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
            
            # Decode and filter events where dataFeedId matches oracle_id
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
                    
                    # Compare hex strings (normalized to lowercase)
                    if event_id_hex == oracle_id_hex:
                        nav_updates.append(decoded_event)
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
    print(f"Found {len(nav_updates)} NAV updates matching oracle_id: {oracle_id}")
    
except Exception as e:
    print(f"Error querying ValueUpdate events: {e}")
    import traceback
    traceback.print_exc()

# Sort by block number (oldest first)
if nav_updates:
    nav_updates.sort(key=lambda x: x.blockNumber)
    
    print(f"\n{'='*80}")
    print(f"NAV UPDATES FOR ORACLE ID: {oracle_id}")
    print(f"{'='*80}\n")
    
    # Display NAV updates
    for i, event in enumerate(nav_updates, 1):
        # Get block timestamp
        try:
            block = w3.eth.get_block(event.blockNumber)
            block_timestamp = block['timestamp']
            dt = datetime.fromtimestamp(block_timestamp)
            timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except:
            timestamp_str = "N/A"
        
        print(f"Update #{i}")
        print(f"  Block Number: {event.blockNumber}")
        print(f"  Block Timestamp: {timestamp_str}")
        print(f"  Transaction Hash: {event.transactionHash.hex()}")
        print(f"  Event: {event.event}")
        print(f"  NAV Value: {event.args.value:,}")
        print(f"  Data Feed ID: {event.args.dataFeedId.hex()}")
        print(f"  Updated At (from event): {event.args.updatedAt}")
        print(f"  Log Index: {event.logIndex}")
        print()
else:
    print(f"\nNo NAV updates found for oracle_id: {oracle_id}")
    print("You may need to adjust the from_block range or check if the contract has emitted events.")

