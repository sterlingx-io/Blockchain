import json
import os
import math
import heapq
import logging
import time
import threading
from Block import Block
from Transaction import Transaction
from BloomFilter import BloomFilter
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from UTXO import UTXO, UTXOSet
import base64
import ecdsa
import hashlib
from SQLiteStorage import SQLiteStorage

class BlockStorage:
    def __init__(self, storage_dir=None):
        """Initialize block storage"""
        if storage_dir is None:
            # Use the specified path in $HOME/.local/share/sterlingx/blocks
            home_dir = os.path.expanduser("~")
            self.storage_dir = os.path.join(home_dir, ".local", "share", "sterlingx", "blocks")
        else:
            self.storage_dir = storage_dir
            
        self.cache = {}  # Block hash -> Block object
        self.cache_size = 1000  # Maximum number of blocks to keep in cache
        self.cache_lock = threading.Lock()  # Lock for cache operations
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            
    def get_block_path(self, block_hash):
        """Get the file path for a block"""
        return os.path.join(self.storage_dir, f"{block_hash}.json")
        
    def save_block(self, block):
        """Save a block to disk"""
        try:
            # Convert block to dictionary
            block_dict = block.to_dict() if hasattr(block, 'to_dict') else block
            
            # Calculate checksum
            block_dict['checksum'] = hashlib.sha256(json.dumps(block_dict).encode()).hexdigest()
            
            # Save to file
            with open(self.get_block_path(block.hash), 'w') as f:
                json.dump(block_dict, f)
                
            # Add to cache
            with self.cache_lock:
                self.cache[block.hash] = block
                if len(self.cache) > self.cache_size:
                    # Remove oldest block from cache
                    self.cache.pop(next(iter(self.cache)))
                
        except Exception as e:
            logging.error(f"Error saving block {block.hash[:8]}: {e}")
            
    def load_block(self, block_hash):
        """Load a block from disk"""
        try:
            # Check cache first
            with self.cache_lock:
                if block_hash in self.cache:
                    return self.cache[block_hash]
                
            # Load from disk
            block_path = self.get_block_path(block_hash)
            if not os.path.exists(block_path):
                return None
                
            with open(block_path, 'r') as f:
                block_dict = json.load(f)
                
            # Verify checksum
            stored_checksum = block_dict.pop('checksum')
            calculated_checksum = hashlib.sha256(json.dumps(block_dict).encode()).hexdigest()
            if stored_checksum != calculated_checksum:
                logging.error(f"Block {block_hash[:8]} checksum mismatch")
                return None
                
            # Convert to Block object
            block = Block.from_dict(block_dict)
            
            # Add to cache
            with self.cache_lock:
                self.cache[block_hash] = block
                if len(self.cache) > self.cache_size:
                    self.cache.pop(next(iter(self.cache)))
                
            return block
            
        except Exception as e:
            logging.error(f"Error loading block {block_hash[:8]}: {e}")
            return None
            
    def delete_block(self, block_hash):
        """Delete a block from storage"""
        try:
            # Remove from cache
            with self.cache_lock:
                self.cache.pop(block_hash, None)
            
            # Delete file
            block_path = self.get_block_path(block_hash)
            if os.path.exists(block_path):
                os.remove(block_path)
                
        except Exception as e:
            logging.error(f"Error deleting block {block_hash[:8]}: {e}")

class Blockchain:
    def __init__(self, storage=None, difficulty_target=2, mining_reward=50, pruning_enabled=True, min_confirmations=100, db_file=None, genesis_address=None):
        """Initialize blockchain"""
        self.storage = storage or SQLiteStorage()
        self.difficulty_target = difficulty_target
        self.mining_reward = mining_reward
        self.utxo_set = UTXOSet()
        self.pruning_enabled = pruning_enabled
        self.min_confirmations = min_confirmations
        self.last_prune_height = 0
        self.pruning_interval = 1000
        self.checkpoint_interval = 100  # Create checkpoints every 100 blocks
        self.checkpoints = {}  # height -> block_hash
        self.last_checkpoint_height = 0
        self.validation_cache = {}  # block_hash -> (is_valid, timestamp)
        self.validation_cache_size = 1000  # Maximum number of validation results to cache
        self.validation_cache_ttl = 3600  # Cache TTL in seconds (1 hour)
        self.db_file = db_file  # Store the database file path
        self.genesis_address = genesis_address  # Store the genesis address
        self.chain = []  # Initialize the blockchain
        self.transaction_pool = []  # Initialize transaction pool
        self.used_txids = set()  # Track used transaction IDs
        self.header_archive = {}  # Store block headers for archived blocks
        self.last_archive_height = 0  # Track last archived block height
        self.archive_interval = 1000  # Archive blocks every 1000 blocks
        self.KEEP_FULL_BLOCKS = 100  # Keep last 100 full blocks
        self.block_size_history = []  # Track block sizes
        self.MAX_BLOCK_SIZE = 1000000  # Maximum block size in bytes
        self.MIN_BLOCK_SIZE = 100  # Minimum block size in bytes
        self.TARGET_BLOCK_SIZE = 500000  # Target block size in bytes
        self.BLOCK_SIZE_ADJUSTMENT_INTERVAL = 2016  # Adjust block size every 2016 blocks
        self.last_block_size_adjustment = 0  # Track last block size adjustment
        self.BURN_ADDRESS = "0000000000000000000000000000DEAD"  # Burn address for fee burning
        
        # Initialize block storage
        self.block_storage = BlockStorage()
        
        # Load from disk if exists
        self._load_from_disk()
        
    def _load_from_disk(self):
        """Load blockchain state from disk"""
        try:
            # Load chain metadata using SQLiteStorage
            metadata = self.storage.load_chain_metadata('chain_metadata')
            if metadata:
                self.difficulty_target = metadata.get('difficulty_target', self.difficulty_target)
                self.mining_reward = metadata.get('mining_reward', self.mining_reward)
                self.pruning_enabled = metadata.get('pruning_enabled', self.pruning_enabled)
                self.min_confirmations = metadata.get('min_confirmations', self.min_confirmations)
                self.last_prune_height = metadata.get('last_prune_height', 0)
                self.checkpoints = metadata.get('checkpoints', {})
                self.last_checkpoint_height = metadata.get('last_checkpoint_height', 0)
                self.transaction_pool = metadata.get('transaction_pool', [])
                self.used_txids = set(metadata.get('used_txids', []))
                self.header_archive = metadata.get('header_archive', {})
                self.last_archive_height = metadata.get('last_archive_height', 0)
                self.block_size_history = metadata.get('block_size_history', [])
                self.last_block_size_adjustment = metadata.get('last_block_size_adjustment', 0)
                
                # Load blocks from chain metadata
                chain_data = metadata.get('chain', [])
                self.chain = []
                for block_dict in chain_data:
                    if isinstance(block_dict, dict):
                        block = Block.from_dict(block_dict)
                        self.chain.append(block)
                
            # Load UTXO set using SQLiteStorage
            utxo_data = self.storage.load_chain_metadata('utxo_set')
            if utxo_data:
                self.utxo_set = UTXOSet.from_dict(utxo_data)
            else:
                self.utxo_set = UTXOSet()
                
            # Load blocks from block storage
            for block in self.chain:
                stored_block = self.block_storage.load_block(block.hash)
                if stored_block:
                    # Update block with data from block storage
                    block = stored_block
                
        except Exception as e:
            logging.error(f"Error loading from disk: {e}")
            # Initialize empty state if loading fails
            self.chain = []
            self.transaction_pool = []
            self.used_txids = set()
            self.utxo_set = UTXOSet()
            self.header_archive = {}
            self.block_size_history = []

    def save_to_disk(self):
        """Save blockchain state to disk"""
        try:
            # Convert blocks to dictionaries
            chain_data = []
            for block in self.chain:
                if hasattr(block, 'to_dict'):
                    chain_data.append(block.to_dict())
                else:
                    chain_data.append(block)
            
            # Save chain metadata using SQLiteStorage
            metadata = {
                'difficulty_target': self.difficulty_target,
                'mining_reward': self.mining_reward,
                'pruning_enabled': self.pruning_enabled,
                'min_confirmations': self.min_confirmations,
                'last_prune_height': self.last_prune_height,
                'checkpoints': self.checkpoints,
                'last_checkpoint_height': self.last_checkpoint_height,
                'chain': chain_data,
                'transaction_pool': self.transaction_pool,
                'used_txids': list(self.used_txids),
                'header_archive': self.header_archive,
                'last_archive_height': self.last_archive_height,
                'block_size_history': self.block_size_history,
                'last_block_size_adjustment': self.last_block_size_adjustment
            }
            self.storage.save_chain_metadata('chain_metadata', metadata)
            
            # Save UTXO set using SQLiteStorage
            self.storage.save_chain_metadata('utxo_set', self.utxo_set.to_dict())
            
            # Save blocks using BlockStorage
            for block in self.chain:
                self.block_storage.save_block(block)
            
        except Exception as e:
            logging.error(f"Error saving to disk: {e}")

    def create_checkpoint(self, block):
        """Create a checkpoint at the given block height"""
        try:
            height = block.index
            if height % self.checkpoint_interval == 0:
                self.checkpoints[height] = block.hash
                self.last_checkpoint_height = height
                logging.info(f"Created checkpoint at height {height}")
                self.save_to_disk()
        except Exception as e:
            logging.error(f"Error creating checkpoint: {e}")
            
    def get_closest_checkpoint(self, height):
        """Get the closest checkpoint below the given height"""
        try:
            # Find the highest checkpoint below the given height
            closest_height = 0
            for checkpoint_height in sorted(self.checkpoints.keys()):
                if checkpoint_height <= height:
                    closest_height = checkpoint_height
                else:
                    break
            return closest_height, self.checkpoints.get(closest_height)
        except Exception as e:
            logging.error(f"Error getting closest checkpoint: {e}")
            return 0, None
            
    def validate_chain(self, start_height=None, end_height=None):
        """Validate the blockchain from a checkpoint or genesis"""
        try:
            # Get the chain length
            chain_length = self.get_chain_length()
            if not chain_length:
                return True
                
            # Determine validation range
            if start_height is None:
                # Find closest checkpoint
                start_height, checkpoint_hash = self.get_closest_checkpoint(chain_length)
                if start_height > 0:
                    logging.info(f"Starting validation from checkpoint at height {start_height}")
                    current_block = self.get_block_by_hash(checkpoint_hash)
                    if not current_block:
                        logging.warning("Checkpoint block not found, starting from genesis")
                        start_height = 0
            else:
                start_height = max(0, start_height)
                
            if end_height is None:
                end_height = chain_length
            else:
                end_height = min(end_height, chain_length)
                
            # Validate blocks in range
            current_height = start_height
            while current_height < end_height:
                current_block = self.get_block_by_height(current_height)
                if not current_block:
                    logging.error(f"Missing block at height {current_height}")
                    return False
                    
                # Validate block
                if not self.validate_block(current_block):
                    logging.error(f"Invalid block at height {current_height}")
                    return False
                    
                # Update UTXO set
                self.utxo_set.update_from_block(current_block)
                
                # Create checkpoint if needed
                if current_height % self.checkpoint_interval == 0:
                    self.create_checkpoint(current_block)
                    
                current_height += 1
                
            return True
            
        except Exception as e:
            logging.error(f"Error validating chain: {e}")
            return False
            
    def add_block(self, block):
        """Add a block to the blockchain"""
        try:
            # Convert block to dictionary if needed
            if hasattr(block, 'to_dict'):
                block_dict = block.to_dict()
            else:
                block_dict = block

            # Validate block
            if not self.validate_block(block_dict):
                return False

            # Add block to chain
            if hasattr(block, 'to_dict'):
                self.chain.append(block)
            else:
                # Convert dictionary to Block object
                block_obj = Block.from_dict(block_dict)
                self.chain.append(block_obj)

            # Update UTXO set
            if not self.update_utxo_set(block):
                # If UTXO update fails, remove block from chain
                self.chain.pop()
                return False

            # Calculate and update block size history
            block_size = self.calculate_block_size(block_dict)
            self.block_size_history.append(block_size)

            # Check if we should prune UTXOs
            current_height = len(self.chain)
            if current_height - self.last_prune_height >= self.pruning_interval:
                self.utxo_set.prune_utxos(current_height)
                self.last_prune_height = current_height

            # Create UTXO set snapshot
            self.utxo_set.create_snapshot(current_height)

            # Check if we should archive blocks
            self.archive_blocks()

            # Save to disk
            self.save_to_disk()

            return True
        except Exception as e:
            logging.error(f"Error adding block: {e}")
            return False

    def calculate_target(self, difficulty_bits):
        """Calculate the difficulty target from difficulty bits"""
        try:
            # Convert difficulty bits to target
            target = 2 ** (256 - difficulty_bits)
            return target
        except Exception as e:
            logging.error(f"Error calculating difficulty target: {e}")
            return 2**256 // 2**16  # Default difficulty

    def calculate_block_size(self, block):
        """Calculate the size of a block in bytes"""
        try:
            # Convert block to JSON and get size
            block_json = json.dumps(block)
            return len(block_json.encode('utf-8'))
        except Exception as e:
            logging.error(f"Error calculating block size: {e}")
            return 0

    def validate_block_size(self, block):
        """Validate block size"""
        try:
            block_size = self.calculate_block_size(block)
            
            # Check if block exceeds maximum size
            if block_size > self.MAX_BLOCK_SIZE:
                logging.error(f"Block size {block_size} exceeds maximum {self.MAX_BLOCK_SIZE}")
                return False
                
            # Check if block is too small
            if block_size < self.MIN_BLOCK_SIZE:
                logging.error(f"Block size {block_size} below minimum {self.MIN_BLOCK_SIZE}")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Error validating block size: {e}")
            return False

    def validate_block_format(self, block):
        """Validate the format of a block"""
        try:
            # Convert block to dictionary if it's a Block object
            if hasattr(block, 'to_dict'):
                block_dict = block.to_dict()
            else:
                block_dict = block

            # Check required fields
            required_fields = ['version', 'previous_hash', 'transactions', 'difficulty_target', 'nonce', 'timestamp', 'hash']
            for field in required_fields:
                if field not in block_dict:
                    logging.error(f"Block missing required field: {field}")
                    return False

            # Validate field types
            if not isinstance(block_dict['version'], int):
                logging.error("Block version must be an integer")
                return False
                
            if not isinstance(block_dict['previous_hash'], str) or len(block_dict['previous_hash']) != 64:
                logging.error("Invalid previous hash format")
                return False
                
            if not isinstance(block_dict['transactions'], list):
                logging.error("Transactions must be a list")
                return False
                
            if not isinstance(block_dict['difficulty_target'], int):
                logging.error("Difficulty target must be an integer")
                return False
                
            if not isinstance(block_dict['nonce'], int):
                logging.error("Nonce must be an integer")
                return False
                
            if not isinstance(block_dict['timestamp'], int):
                logging.error("Timestamp must be an integer")
                return False
                
            if not isinstance(block_dict['hash'], str) or len(block_dict['hash']) != 64:
                logging.error("Invalid block hash format")
                return False

            # Special handling for genesis block
            if block_dict['previous_hash'] == "0" * 64:
                # Genesis block must have exactly one transaction
                if len(block_dict['transactions']) != 1:
                    logging.error("Genesis block must have exactly one transaction")
                    return False
                    
                # Genesis transaction must be valid
                genesis_tx = block_dict['transactions'][0]
                if not self.validate_transaction_format(genesis_tx):
                    logging.debug("Invalid genesis transaction format")
                    return False
                    
                # Genesis transaction must be of type 'genesis'
                if genesis_tx.get('type') != 'genesis':
                    logging.error("Genesis transaction must be of type 'genesis'")
                    return False
                    
                # Genesis block must have index 0
                if block_dict.get('index') != 0:
                    logging.error("Genesis block must have index 0")
                    return False
                    
                # Genesis block must have timestamp 0
                if block_dict['timestamp'] != 0:
                    logging.error("Genesis block must have timestamp 0")
                    return False
                    
                return True

            # Validate transaction format for non-genesis blocks
            for tx in block_dict['transactions']:
                if not self.validate_transaction_format(tx):
                    return False

            return True
            
        except Exception as e:
            logging.error(f"Error validating block format: {e}")
            return False

    def verify_block_hash(self, block):
        """Verify that the block's hash is valid and matches its contents"""
        try:
            # Recalculate the block hash
            block_copy = Block(
                version=block.version,
                previous_hash=block.previous_hash,
                transactions=block.transactions,
                difficulty_target=block.difficulty_target,
                nonce=block.nonce,
                timestamp=block.timestamp,
                index=block.index if hasattr(block, 'index') else 0
            )
            block_copy.update_hash()
            
            # Compare with stored hash
            if block_copy.hash != block.hash:
                logging.error(f"Block hash mismatch. Expected: {block.hash}, Got: {block_copy.hash}")
                return False
                
            # Skip proof of work check for genesis block
            if block.previous_hash == "0" * 64:  # This identifies the genesis block
                return True
                
            # Verify proof of work for non-genesis blocks
            target = self.calculate_target(block.difficulty_target)
            block_hash_int = int(block.hash, 16)
            if block_hash_int >= target:
                logging.error(f"Block hash {block.hash} does not meet difficulty target")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying block hash: {e}")
            return False

    def verify_block_signature(self, block):
        """Verify the signature of a block"""
        try:
            # Skip signature verification for genesis block
            if block.previous_hash == "0" * 64:  # This identifies the genesis block
                return True
                
            # Skip signature verification for blocks without a signature field
            if not hasattr(block, 'signature') or not hasattr(block, 'public_key'):
                return True
                
            # Create message to verify
            message = f"{block.version}{block.previous_hash}{block.transactions}{block.difficulty_target}{block.nonce}{block.timestamp}".encode()
            
            try:
                # Load public key
                public_key = serialization.load_pem_public_key(
                    block.public_key.encode(),
                    backend=default_backend()
                )
                
                # Verify signature
                public_key.verify(
                    bytes.fromhex(block.signature),
                    message,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                return True
            except Exception as e:
                logging.error(f"Block signature verification failed: {e}")
                return False
                
        except Exception as e:
            logging.error(f"Error verifying block signature: {e}")
            return False

    def verify_block_difficulty(self, block):
        """Verify that the block's difficulty meets the target"""
        try:
            # Skip difficulty check for genesis block
            if block.previous_hash == "0" * 64:  # This identifies the genesis block
                return True
                
            # Calculate target from difficulty bits
            target = self.calculate_target(block.difficulty_target)
            
            # Convert block hash to integer
            block_hash_int = int(block.hash, 16)
            
            # Verify hash is less than target
            if block_hash_int >= target:
                logging.error(f"Block hash {block.hash} does not meet difficulty target")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error verifying block difficulty: {e}")
            return False

    def adjust_block_size_limit(self):
        """Adjust block size limit based on network conditions"""
        try:
            # Only adjust every BLOCK_SIZE_ADJUSTMENT_INTERVAL blocks
            current_height = len(self.chain)
            if current_height - self.last_block_size_adjustment < self.BLOCK_SIZE_ADJUSTMENT_INTERVAL:
                return
                
            # Calculate average block size
            if len(self.block_size_history) < self.BLOCK_SIZE_HISTORY_SIZE:
                return
                
            avg_block_size = sum(self.block_size_history) / len(self.block_size_history)
            
            # Adjust block size limit
            if avg_block_size > self.TARGET_BLOCK_SIZE * 0.95:  # 95% of target
                # Increase block size limit by 10%
                new_limit = min(self.MAX_BLOCK_SIZE, int(self.MAX_BLOCK_SIZE * 1.1))
                self.MAX_BLOCK_SIZE = new_limit
                logging.info(f"Block size limit increased to {new_limit}")
            elif avg_block_size < self.TARGET_BLOCK_SIZE * 0.5:  # 50% of target
                # Decrease block size limit by 10%
                new_limit = max(self.MIN_BLOCK_SIZE, int(self.MAX_BLOCK_SIZE * 0.9))
                self.MAX_BLOCK_SIZE = new_limit
                logging.info(f"Block size limit decreased to {new_limit}")
                
            # Update adjustment height
            self.last_block_size_adjustment = current_height
            
        except Exception as e:
            logging.error(f"Error adjusting block size limit: {e}")

    def add_block(self, block):
        """Add a block to the blockchain"""
        try:
            # Convert block to dictionary if needed
            if hasattr(block, 'to_dict'):
                block_dict = block.to_dict()
            else:
                block_dict = block

            # Validate block
            if not self.validate_block(block_dict):
                return False

            # Add block to chain
            if hasattr(block, 'to_dict'):
                self.chain.append(block)
            else:
                # Convert dictionary to Block object
                block_obj = Block.from_dict(block_dict)
                self.chain.append(block_obj)

            # Update UTXO set
            if not self.update_utxo_set(block):
                # If UTXO update fails, remove block from chain
                self.chain.pop()
                return False

            # Calculate and update block size history
            block_size = self.calculate_block_size(block_dict)
            self.block_size_history.append(block_size)

            # Check if we should prune UTXOs
            current_height = len(self.chain)
            if current_height - self.last_prune_height >= self.pruning_interval:
                self.utxo_set.prune_utxos(current_height)
                self.last_prune_height = current_height

            # Create UTXO set snapshot
            self.utxo_set.create_snapshot(current_height)

            # Check if we should archive blocks
            self.archive_blocks()

            # Save to disk
            self.save_to_disk()

            return True
        except Exception as e:
            logging.error(f"Error adding block: {e}")
            return False

    def update_utxo_set(self, block):
        """Update UTXO set with transactions from a block"""
        try:
            current_height = len(self.chain)
            for tx in block.transactions:
                # Convert to dictionary if it's a Transaction object
                tx_dict = tx.to_dict() if hasattr(tx, 'to_dict') else tx
                
                # Skip if transaction already processed
                if tx_dict['txid'] in self.used_txids:
                    logging.debug(f"Transaction {tx_dict['txid'][:8]} already processed, skipping")
                    continue
                
                # Handle genesis transaction
                if tx_dict.get('sender_address') == 'GENESIS':
                    for i, output in enumerate(tx_dict['outputs']):
                        utxo_id = f"{tx_dict['txid']}:{i}"
                        existing_utxo = self.utxo_set.get_utxo(utxo_id)
                        if not existing_utxo:
                            utxo = UTXO(tx_dict['txid'], i, output['amount'], output['address'])
                            self.utxo_set.add_utxo(utxo)
                            logging.info(f"Added genesis UTXO {utxo.get_id()} with amount {output['amount']} for {output['address']}")
                    self.used_txids.add(tx_dict['txid'])
                    continue
                    
                # Handle coinbase transaction
                if tx_dict.get('sender_address') == 'COINBASE':
                    for i, output in enumerate(tx_dict['outputs']):
                        utxo_id = f"{tx_dict['txid']}:{i}"
                        existing_utxo = self.utxo_set.get_utxo(utxo_id)
                        if not existing_utxo:
                            utxo = UTXO(tx_dict['txid'], i, output['amount'], output['address'])
                            self.utxo_set.add_utxo(utxo)
                            logging.debug(f"Added coinbase UTXO {utxo.get_id()} with amount {output['amount']} for {output['address']}")
                    self.used_txids.add(tx_dict['txid'])
                    continue
                    
                # Handle regular transactions
                # First check all inputs exist and are unspent
                for input_tx in tx_dict['inputs']:
                    utxo_id = f"{input_tx['txid']}:{input_tx['output_index']}"
                    utxo = self.utxo_set.get_utxo(utxo_id)
                    if not utxo or utxo.spent:
                        logging.error(f"Input UTXO {utxo_id} not found or already spent")
                        return False
                        
                # Mark inputs as spent
                for input_tx in tx_dict['inputs']:
                    utxo_id = f"{input_tx['txid']}:{input_tx['output_index']}"
                    current_height = len(self.chain)
                    self.utxo_set.spend_utxo(utxo_id, current_height)
                    logging.info(f"Spent UTXO {utxo_id} for transaction {tx_dict['txid'][:8]}")
                        
                # Add new outputs
                for i, output in enumerate(tx_dict['outputs']):
                    utxo_id = f"{tx_dict['txid']}:{i}"
                    existing_utxo = self.utxo_set.get_utxo(utxo_id)
                    if not existing_utxo:
                        utxo = UTXO(tx_dict['txid'], i, output['amount'], output['address'])
                        self.utxo_set.add_utxo(utxo)
                        logging.info(f"Added UTXO {utxo.get_id()} with amount {output['amount']} for {output['address']}")
                        
                self.used_txids.add(tx_dict['txid'])
            
            return True
        except Exception as e:
            logging.error(f"Error updating UTXO set: {e}")
            return False

    def get_pruning_stats(self):
        """Get statistics about UTXO pruning"""
        return self.utxo_set.get_pruning_stats()

    def get_balance(self, address):
        """Get balance for an address"""
        try:
            # Get all UTXOs for the address
            utxos = self.utxo_set.get_address_utxos(address)
            
            # Calculate total balance
            total = 0.0
            for utxo in utxos:
                if not utxo.spent:
                    total += float(utxo.amount)
                    
            logging.info(f"Balance for {address}: {total}")
            return total
        except Exception as e:
            logging.error(f"Error getting balance: {e}")
            return 0.0

    def add_transaction(self, transaction):
        """Add a transaction to the pool"""
        try:
            # Validate transaction
            if not self.validate_transaction(transaction):
                return False

            with self.pool_lock:
                # Check if transaction already in pool
                if transaction.txid in self.transaction_pool:
                    return False

                # Add to pool
                self.transaction_pool.append(transaction.txid)
                return True

        except Exception as e:
            self.logger.error(f"Error adding transaction: {e}")
            return False

    def check_for_forks(self):
        """Check for chain forks and handle them"""
        try:
            # Get current chain tip
            current_tip = self.chain[-1].hash if self.chain else None
            
            # Check for competing chains
            for block_hash, block in self.orphan_blocks.items():
                # Build chain from orphan block
                chain = self.build_chain_from_block(block)
                if not chain:
                    continue
                    
                # Store competing chain
                self.fork_chains[block_hash] = chain
                
            # Resolve forks if needed
            if len(self.fork_chains) > 0:
                self.resolve_forks()
                
        except Exception as e:
            logging.error(f"Error checking for forks: {e}")

    def _reorganize_chain(self, new_chain):
        """Reorganize the blockchain to a new chain"""
        try:
            # Find the fork point
            fork_point = 0
            for i in range(min(len(self.chain), len(new_chain))):
                if self.chain[i].hash != new_chain[i].hash:
                    fork_point = i
                    break
                
            # Roll back transactions from old chain
            for i in range(len(self.chain) - 1, fork_point - 1, -1):
                block = self.chain[i]
                for tx in reversed(block.transactions):
                    # Skip coinbase transactions
                    if tx.sender_address == "COINBASE":
                        continue
                        
                    # Remove outputs from UTXO set
                    for output in tx.outputs:
                        utxo_id = f"{tx.txid}:{output['address']}"
                        if utxo_id in self.utxo_set:
                            del self.utxo_set[utxo_id]
                            
                    # Add inputs back to UTXO set
                    for input_tx in tx.inputs:
                        utxo_id = f"{input_tx['txid']}:{input_tx['address']}"
                        self.utxo_set[utxo_id] = {
                            'amount': input_tx['amount'],
                            'address': input_tx['address']
                        }
                        
            # Apply transactions from new chain
            current_height = fork_point
            for i in range(fork_point, len(new_chain)):
                current_height += 1
                block = new_chain[i]
                for tx in block.transactions:
                    # Skip coinbase transactions
                    if tx.sender_address == "COINBASE":
                        continue
                        
                    # Remove inputs from UTXO set
                    for input_tx in tx.inputs:
                        utxo_id = f"{input_tx['txid']}:{input_tx['address']}"
                        if utxo_id in self.utxo_set:
                            self.utxo_set.spend_utxo(utxo_id, current_height)
                            
                    # Add outputs to UTXO set
                    for output in tx.outputs:
                        utxo_id = f"{tx.txid}:{output['address']}"
                        self.utxo_set.add_utxo(UTXO(
                            txid=tx.txid,
                            output_index=output['index'],
                            amount=output['amount'],
                            address=output['address']
                        ))
                        
            # Update chain
            self.chain = new_chain
            
            # Save to disk
            if self.db_file:
                self.save_to_db()
                
            logging.info(f"Reorganized chain to new chain with height {len(new_chain)}")
            return True
        except Exception as e:
            logging.error(f"Error reorganizing chain: {e}")
            return False

    def calculate_total_work(self, chain=None):
        """Calculate the total work of a chain.
        
        Args:
            chain: Optional chain to calculate work for. If None, uses self.chain.
        """
        if chain is None:
            chain = self.chain
        
        total_work = 0
        for block in chain:
            target = 2 ** (256 - block.difficulty_target)
            work = (2 ** 256) // target
            total_work += work
        return total_work

    def calculate_chain_work(self, chain):
        """Calculate the total work of a given chain."""
        return self.calculate_total_work(chain)

    def validate_block(self, block):
        """Validate a block"""
        try:
            # Convert block to dictionary if it's a Block object
            if hasattr(block, 'to_dict'):
                block_dict = block.to_dict()
            else:
                block_dict = block

            # Check block format
            if not isinstance(block_dict, dict):
                logging.error("Block is not a dictionary")
                return False
            
            # Check required fields
            required_fields = ['version', 'previous_hash', 'transactions', 'difficulty_target', 'nonce', 'timestamp', 'hash', 'index', 'merkle_root']
            for field in required_fields:
                if field not in block_dict:
                    logging.error(f"Block missing required field: {field}")
                    return False
            
            # Check block index
            if not isinstance(block_dict['index'], int) or block_dict['index'] < 0:
                logging.error(f"Invalid block index: {block_dict['index']}")
                return False
            
            # Check timestamp
            if block_dict['previous_hash'] == '0' * 64:  # Genesis block
                if block_dict['timestamp'] != 0:
                    logging.error("Genesis block must have timestamp 0")
                    return False
            else:  # Non-genesis block
                if not isinstance(block_dict['timestamp'], (int, float)) or block_dict['timestamp'] <= 0:
                    logging.error(f"Invalid block timestamp: {block_dict['timestamp']}")
                    return False
            
            # Check transactions
            if not isinstance(block_dict['transactions'], list):
                logging.error("Block transactions is not a list")
                return False
            
            # Check previous hash
            if not isinstance(block_dict['previous_hash'], str) or len(block_dict['previous_hash']) != 64:
                logging.error(f"Invalid previous hash: {block_dict['previous_hash']}")
                return False
            
            # Check current hash
            if not isinstance(block_dict['hash'], str) or len(block_dict['hash']) != 64:
                logging.error(f"Invalid block hash: {block_dict['hash']}")
                return False
            
            # Check nonce
            if not isinstance(block_dict['nonce'], int) or block_dict['nonce'] < 0:
                logging.error(f"Invalid block nonce: {block_dict['nonce']}")
                return False
            
            # Check difficulty target
            if not isinstance(block_dict['difficulty_target'], int) or block_dict['difficulty_target'] <= 0:
                logging.error(f"Invalid difficulty target: {block_dict['difficulty_target']}")
                return False
            
            # Special handling for genesis block
            if block_dict['previous_hash'] == '0' * 64:
                # Genesis block must have exactly one transaction
                if len(block_dict['transactions']) != 1:
                    logging.error("Genesis block must have exactly one transaction")
                    return False
                    
                # Genesis transaction must be valid
                genesis_tx = block_dict['transactions'][0]
                if not self.validate_transaction_format(genesis_tx):
                    logging.debug("Invalid genesis transaction format")
                    return False
                    
                # Genesis transaction must be of type 'genesis'
                if genesis_tx.get('type') != 'genesis':
                    logging.error("Genesis transaction must be of type 'genesis'")
                    return False
                    
                # Genesis block must have index 0
                if block_dict['index'] != 0:
                    logging.error("Genesis block must have index 0")
                    return False
                    
                return True
            
            # Create a Block object to verify the hash
            temp_block = Block(
                version=block_dict.get('version', 1),
                previous_hash=block_dict['previous_hash'],
                transactions=block_dict['transactions'],
                difficulty_target=block_dict['difficulty_target'],
                nonce=block_dict['nonce'],
                timestamp=block_dict['timestamp'],
                index=block_dict['index']
            )
            temp_block.update_hash()
            
            # Verify the hash matches
            if temp_block.hash != block_dict['hash']:
                logging.error(f"Block hash mismatch: expected {block_dict['hash']}, got {temp_block.hash}")
                return False
            
            # Verify proof of work
            if not self.check_proof_of_work(block_dict):
                logging.error("Block proof of work is invalid")
                return False
            
            # Verify transactions
            for tx in block_dict['transactions']:
                if not self.validate_transaction_format(tx):
                    logging.error(f"Invalid transaction format in block: {tx['txid'][:8]}")
                    return False
                
                # Skip signature verification for special transaction types
                if tx['sender_address'] not in ["COINBASE", "FEES", "GENESIS"]:
                    if not self.verify_transaction_signature(tx):
                        logging.error(f"Invalid transaction signature in block: {tx['txid'][:8]}")
                        return False
                
                # Skip fee validation for special transaction types
                if tx['sender_address'] not in ["COINBASE", "FEES", "GENESIS"]:
                    if not self.validate_transaction_fee(tx):
                        logging.error(f"Invalid transaction fee in block: {tx['txid'][:8]}")
                        return False
            
            return True
        except Exception as e:
            logging.error(f"Error validating block: {e}")
            return False

    def check_proof_of_work(self, block):
        """Check if the block's hash meets the difficulty target"""
        try:
            # Convert block hash to integer
            hash_int = int(block['hash'], 16)
            
            # Calculate target
            target = 2**256 - block['difficulty_target']
            
            # Check if hash is less than target
            return hash_int < target
        except Exception as e:
            logging.error(f"Error checking proof of work: {e}")
            return False

    def validate_transactions(self, transactions, check_used_txids=True):
        """Validate a list of transactions"""
        try:
            # Check for duplicate transaction IDs
            txids = set()
            for tx in transactions:
                if tx["txid"] in txids:
                    logging.error("❌ Invalid transactions: Duplicate transaction ID")
                    return False
                txids.add(tx["txid"])
                
            # Check for duplicate transaction IDs in used_txids if requested
            if check_used_txids:
                for tx in transactions:
                    if tx["txid"] in self.used_txids:
                        logging.error("❌ Invalid transactions: Transaction ID already used")
                        return False
                    
            # Check for coinbase transaction
            if not transactions or not self.is_coinbase_tx(transactions[0]):
                logging.error("❌ Invalid transactions: Missing coinbase transaction")
                return False
                
            # Check other transactions
            for tx in transactions[1:]:
                if not self.validate_transaction(tx):
                    logging.error(f"❌ Invalid transactions: Transaction {tx['txid'][:8]} is invalid")
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Error in validate_transactions: {e}")
            return False
            
    def validate_transaction(self, tx):
        """Validate a transaction"""
        try:
            # Skip validation for coinbase, genesis, and fee burn transactions
            if self.is_coinbase_tx(tx) or tx.get('sender_address') == "GENESIS" or tx.get('type') == "feeburn":
                return True

            # Prevent spending from burn address
            for input_tx in tx["inputs"]:
                utxo_id = f"{input_tx['txid']}:{input_tx['output_index']}"
                utxo = self.utxo_set.get_utxo(utxo_id)
                if utxo and utxo.address == self.BURN_ADDRESS:
                    logging.error("Cannot spend from burn address")
                    return False

            # Check required fields
            required_fields = ['txid', 'sender_address', 'receiver', 'amount', 'fee', 'timestamp', 'outputs', 'signature']
            for field in required_fields:
                if field not in tx:
                    logging.error(f"Missing required field: {field}")
                    return False

            # Validate sender address format
            if not isinstance(tx['sender_address'], str) or len(tx['sender_address']) != 32:
                logging.error(f"Invalid sender address format: {tx['sender_address']}")
                return False

            # Validate receiver address format
            if not isinstance(tx['receiver'], str) or len(tx['receiver']) != 32:
                logging.error(f"Invalid receiver address format: {tx['receiver']}")
                return False

            # Validate amounts
            if not isinstance(tx['amount'], (int, float)) or tx['amount'] <= 0:
                logging.error(f"Invalid amount: {tx['amount']}")
                return False

            if not isinstance(tx['fee'], (int, float)) or tx['fee'] < 0:
                logging.error(f"Invalid fee: {tx['fee']}")
                return False

            # Validate outputs
            if not isinstance(tx['outputs'], list) or not tx['outputs']:
                logging.error("Invalid outputs format")
                return False

            for output in tx['outputs']:
                if not isinstance(output, dict):
                    logging.error("Invalid output format")
                    return False
                if 'address' not in output or 'amount' not in output:
                    logging.error("Missing required output fields")
                    return False
                if not isinstance(output['address'], str) or len(output['address']) != 32:
                    logging.error(f"Invalid output address format: {output['address']}")
                    return False
                if not isinstance(output['amount'], (int, float)) or output['amount'] <= 0:
                    logging.error(f"Invalid output amount: {output['amount']}")
                    return False

            # Verify inputs
            if not self.verify_transaction_inputs(tx):
                logging.debug("Transaction inputs verification failed")
                return False

            # Verify signature
            if not self.verify_transaction_signature(tx):
                logging.error("Transaction signature verification failed")
                return False

            logging.info(f"Transaction {tx['txid'][:8]} validated successfully")
            return True
        except Exception as e:
            logging.error(f"Error validating transaction: {e}")
            return False

    def is_double_spend(self, txid, output_index):
        """Check if UTXO is already spent"""
        try:
            # Create UTXO ID
            utxo_id = f"{txid}:{output_index}"
            
            # Check if UTXO exists and is unspent
            utxo = self.utxo_set.get_utxo(utxo_id)
            if not utxo or utxo.spent:
                return True
                
            # Check if UTXO is already spent in mempool
            for tx in self.transaction_pool:
                for input_tx in tx['inputs']:
                    if input_tx['txid'] == txid and input_tx['output_index'] == output_index:
                        return True
                        
            return False
        except Exception as e:
            logging.error(f"Error checking double spend: {e}")
            return True

    def has_conflicting_transactions(self, transaction):
        """Check if transaction conflicts with existing transactions"""
        try:
            # Check conflicts in mempool
            for tx in self.transaction_pool:
                if tx.txid == transaction.txid:
                    continue
                    
                # Check if transactions share any inputs
                for input_tx in transaction.inputs:
                    for other_input in tx.inputs:
                        if input_tx['txid'] == other_input['txid'] and input_tx['output_index'] == other_input['output_index']:
                            return True
                            
            return False
        except Exception as e:
            logging.error(f"Error checking transaction conflicts: {e}")
            return True

    def is_coinbase_tx(self, tx):
        """Check if a transaction is a coinbase transaction"""
        try:
            logging.info(f"Checking if tx is coinbase: {tx.get('txid', 'no-txid')[:8]} - {tx.get('sender_address')}")
            
            is_coinbase = (
                isinstance(tx, dict) and
                tx.get("sender_address") == "COINBASE" and
                "outputs" in tx and
                len(tx.get("outputs", [])) > 0
            )
            
            return is_coinbase
        except Exception as e:
            logging.error(f"Error checking coinbase transaction: {e}")
            return False

    def update_balances(self, block):
        """Update balances for all addresses in a block"""
        try:
            current_height = len(self.chain)
            processed_utxos = set()
            
            for tx in block.transactions:
                # Handle coinbase and genesis transactions
                if tx["sender_address"] in ["COINBASE", "GENESIS"]:
                    # Add output to UTXO set
                    for i, output in enumerate(tx["outputs"]):
                        utxo_id = f"{tx['txid']}:{i}"
                        if utxo_id not in processed_utxos:
                            utxo = UTXO(
                                txid=tx["txid"],
                                output_index=i,
                                amount=float(output["amount"]),
                                address=output["address"]
                            )
                            self.utxo_set.add_utxo(utxo)
                            processed_utxos.add(utxo_id)
                            logging.info(f"Added UTXO {utxo.get_id()} with amount {output['amount']} for {tx['sender_address']} transaction")
                    continue
                    
                # Handle regular transactions
                # First check all inputs exist and are unspent
                for input_tx in tx["inputs"]:
                    utxo_id = f"{input_tx['txid']}:{input_tx['output_index']}"
                    utxo = self.utxo_set.get_utxo(utxo_id)
                    if not utxo or utxo.spent:
                        logging.debug(f"Input UTXO {utxo_id} not found or already spent")
                        return False
                        
                # Mark inputs as spent
                for input_tx in tx["inputs"]:
                    utxo_id = f"{input_tx['txid']}:{input_tx['output_index']}"
                    current_height = len(self.chain)
                    self.utxo_set.spend_utxo(utxo_id, current_height)
                    logging.info(f"Spent UTXO {utxo_id} for transaction {tx['txid'][:8]}")
                            
                # Add to outputs
                for i, output in enumerate(tx["outputs"]):
                    output_address = output["address"]
                    amount = output["amount"]
                    utxo_id = f"{tx['txid']}:{i}"
                    if utxo_id not in processed_utxos:
                        utxo = UTXO(
                            txid=tx["txid"],
                            output_index=i,
                            amount=float(amount),
                            address=output_address
                        )
                        self.utxo_set.add_utxo(utxo)
                        processed_utxos.add(utxo_id)
                        logging.info(f"Added UTXO {utxo.get_id()} with amount {amount}")

                # Remove from transaction pool if present
                self.transaction_pool = [t for t in self.transaction_pool if t["txid"] != tx["txid"]]

            return True
        except Exception as e:
            logging.debug(f"Error updating balances: {e}")
            return False

    def get_balance(self, address):
        """Get balance for an address"""
        try:
            # Get all UTXOs for the address
            utxos = self.utxo_set.get_address_utxos(address)
            
            # Calculate total balance
            total = 0.0
            for utxo in utxos:
                if not utxo.spent:
                    total += float(utxo.amount)
                    
            logging.info(f"Balance for {address}: {total}")
            return total
        except Exception as e:
            logging.error(f"Error getting balance: {e}")
            return 0.0

    def add_to_pool(self, transaction_data):
        """Add a transaction to the pool if valid"""
        try:
            # Check if transaction is already in pool or used
            if transaction_data["txid"] in self.used_txids:
                logging.debug(f"Transaction {transaction_data['txid']} already used")
                return False

            # Check if transaction is already in pool
            for tx in self.transaction_pool:
                if tx["txid"] == transaction_data["txid"]:
                    logging.debug(f"Transaction {transaction_data['txid']} already in pool")
                    return False

            # Validate transaction format
            if not self.validate_transaction_format(transaction_data):
                logging.debug(f"Invalid transaction format for {transaction_data['txid']}")
                return False

            # Handle genesis transaction
            if transaction_data["sender_address"] == "GENESIS":
                # Add output to UTXO set
                for i, output in enumerate(transaction_data["outputs"]):
                    utxo = UTXO(
                        txid=transaction_data["txid"],
                        output_index=i,
                        amount=float(output["amount"]),
                        address=output["address"]
                    )
                    self.utxo_set.add_utxo(utxo)
                self.transaction_pool.append(transaction_data)
                return True

            # Check for double-spending in the pool
            for tx in self.transaction_pool:
                for input_tx in transaction_data["inputs"]:
                    for existing_input in tx["inputs"]:
                        if (input_tx["txid"] == existing_input["txid"] and 
                            input_tx["output_index"] == existing_input["output_index"]):
                            logging.error(f"Double-spend attempt detected: UTXO {input_tx['txid']}:{input_tx['output_index']} is already being spent in transaction {tx['txid']}")
                            return False

            # Validate transaction using UTXO model
            if not self.validate_transaction(transaction_data):
                logging.debug(f"Invalid transaction {transaction_data['txid']}")
                return False

            # Add to pool and sort by fee
            self.transaction_pool.append(transaction_data)
            self.sort_transaction_pool()
            logging.debug(f"Added transaction {transaction_data['txid']} to pool")
            return True
        except Exception as e:
            logging.error(f"Error adding transaction to pool: {e}")
            return False

    def validate_transaction_fee(self, tx):
        """Validate a transaction fee"""
        try:
            # Skip validation for coinbase and genesis transactions
            if tx['sender_address'] == "COINBASE" or tx['sender_address'] == "GENESIS":
                return True
            
            # Create a temporary transaction object for verification
            temp_tx = Transaction(
                inputs=tx['inputs'],
                outputs=tx['outputs'],
                fee=tx['fee'],
                nonce=tx['nonce'],
                timestamp=tx['timestamp'],
                network_id=tx.get('network_id'),
                chain_id=tx.get('chain_id'),
                sender_address=tx['sender_address'],
                public_key=tx.get('public_key'),
                signature=tx.get('signature')
            )
            
            # Calculate total input amount
            total_input = 0
            for input_tx in tx['inputs']:
                utxo_id = f"{input_tx['txid']}:{input_tx['output_index']}"
                utxo = self.utxo_set.get_utxo(utxo_id)
                if utxo:
                    total_input += utxo.amount  # Access amount as an attribute
                
            # Calculate total output amount
            total_output = 0
            for output in tx['outputs']:
                total_output += output['amount']
            
            # Check if fee is valid
            if total_input < total_output + tx['fee']:
                logging.error(f"Invalid fee: input {total_input} less than output {total_output} plus fee {tx['fee']}")
                return False
            
            # Check if fee is too low - reduced minimum fee to 0.001
            min_fee = 0.001  # Minimum fee in satoshis
            if tx['fee'] < min_fee:
                logging.error(f"Fee too low: {tx['fee']} less than minimum {min_fee}")
                return False
            
            return True
        except Exception as e:
            logging.error(f"Error validating transaction fee: {e}")
            return False

    def sort_transaction_pool(self):
        """Sort transaction pool by fee in descending order"""
        self.transaction_pool.sort(key=lambda x: x["fee"], reverse=True)

    def get_transactions_for_block(self, max_size=1000000):
        """Get transactions for a new block"""
        try:
            logging.info(f"Getting transactions for block, pool size: {len(self.transaction_pool)}")
            
            # Create coinbase transaction
            coinbase_tx = Transaction(
                inputs=[],
                outputs=[{
                    "address": self.genesis_address or "0" * 32,
                    "amount": 1.0
                }],
                fee=1,
                timestamp=int(time.time()),
                sender_address="COINBASE"
            )
            
            # Create coinbase transaction dictionary
            coinbase_dict = {
                "txid": coinbase_tx.txid,
                "sender_address": "COINBASE",
                "inputs": [],  # Coinbase transactions have no inputs
                "outputs": coinbase_tx.outputs,
                "fee": 1,  # Must be a valid fee
                "timestamp": coinbase_tx.timestamp,
                "public_key": "COINBASE",
                "signature": "COINBASE" * 16,  # Must be 16 repetitions
                "nonce": 0
            }
            
            selected_transactions = [coinbase_dict]
            current_size = len(json.dumps(coinbase_dict).encode('utf-8'))
            logging.info(f"Added coinbase transaction, size: {current_size}")

            # Sort pool by fee if not already sorted
            self.sort_transaction_pool()
            logging.info(f"Sorted transaction pool by fee")

            # Add regular transactions
            pool_transactions_added = 0
            failed_transactions = []
            for tx in self.transaction_pool:
                # Skip genesis transactions as they are handled separately
                if tx["sender_address"] == "GENESIS":
                    continue
                    
                # Calculate transaction size
                tx_size = len(json.dumps(tx).encode('utf-8'))
                
                # Check if adding this transaction would exceed block size
                if current_size + tx_size > max_size:
                    logging.info(f"Block size limit reached: {current_size}/{max_size}")
                    break
                
                # Validate transaction before adding
                if not self.validate_transaction(tx):
                    logging.debug(f"Transaction {tx['txid'][:8]} failed validation")
                    failed_transactions.append(tx)
                    continue
                    
                selected_transactions.append(tx)
                current_size += tx_size
                pool_transactions_added += 1
                logging.info(f"Added transaction {tx['txid'][:8]} to block, size: {tx_size}, fee: {tx['fee']}")

            # Remove failed transactions from pool
            for tx in failed_transactions:
                self.transaction_pool.remove(tx)
                logging.debug(f"Removed failed transaction {tx['txid'][:8]} from pool")

            # Add genesis transactions at the end
            genesis_transactions_added = 0
            for tx in self.transaction_pool:
                if tx["sender_address"] == "GENESIS":
                    tx_size = len(json.dumps(tx).encode('utf-8'))
                    if current_size + tx_size > max_size:
                        break
                    selected_transactions.append(tx)
                    current_size += tx_size
                    genesis_transactions_added += 1
                    logging.info(f"Added genesis transaction {tx['txid'][:8]} to block")

            logging.info(f"Selected {len(selected_transactions)} transactions for block (1 coinbase, {pool_transactions_added} regular, {genesis_transactions_added} genesis)")
            return selected_transactions
        except Exception as e:
            logging.error(f"Error getting transactions for block: {e}")
            return []

    def validate_transaction_format(self, tx):
        """Validate transaction format"""
        try:
            # Check required fields
            required_fields = ['txid', 'sender_address', 'timestamp', 'inputs', 'outputs', 'signature']
            for field in required_fields:
                if field not in tx:
                    logging.error(f"Transaction missing required field: {field}")
                    return False
                
            # Check inputs and outputs
            if not isinstance(tx['inputs'], list):
                logging.error("Transaction inputs must be a list")
                return False
            
            if not isinstance(tx['outputs'], list):
                logging.error("Transaction outputs must be a list")
                return False
            
            # Check input format
            for input_tx in tx['inputs']:
                if not all(k in input_tx for k in ['txid', 'output_index', 'amount', 'address']):
                    logging.error("Invalid input format")
                    return False
                
            # Check output format
            for output in tx['outputs']:
                if not all(k in output for k in ['address', 'amount']):
                    logging.error("Invalid output format")
                    return False
            
            # Check string fields
            if not isinstance(tx['txid'], str) or len(tx['txid']) != 64:
                logging.error(f"Invalid txid: {tx['txid']}")
                return False
            
            # Special handling for coinbase and genesis transactions
            if tx['sender_address'] in ['COINBASE', 'GENESIS']:
                if tx['sender_address'] == 'COINBASE' and tx['signature'] != "COINBASE" * 16:
                    logging.error("Invalid coinbase signature")
                    return False
                elif tx['sender_address'] == 'GENESIS' and tx['signature'] != "0" * 128:
                    logging.debug("Invalid genesis signature")
                    return False
                return True
            
            return True
        except Exception as e:
            logging.error(f"Error validating transaction format: {e}")
            return False

    def verify_transaction_signature(self, tx):
        """Verify a transaction signature"""
        try:
            # Skip verification for coinbase and genesis transactions
            if tx['sender_address'] in ["COINBASE", "GENESIS"]:
                return True
                
            # Check signature length
            signature = bytes.fromhex(tx['signature'])
            if len(signature) != 64:
                logging.error(f"Invalid signature length: {len(signature)} bytes")
                return False
            
            # Create transaction dictionary for verification (without signature)
            tx_dict = {
                'txid': tx['txid'],
                'sender_address': tx['sender_address'],
                'receiver': tx['receiver'],
                'amount': tx['amount'],
                'fee': tx['fee'],
                'inputs': tx['inputs'],
                'outputs': tx['outputs'],
                'timestamp': tx['timestamp'],
                'nonce': tx['nonce'],
                'network_id': tx['network_id'],
                'chain_id': tx['chain_id'],
                'rbf': tx['rbf'],
                'rbf_attempt': tx['rbf_attempt']
            }
            
            # Convert to bytes
            tx_bytes = json.dumps(tx_dict, sort_keys=True).encode()
            
            # Verify signature using raw format
            public_key = ecdsa.VerifyingKey.from_string(bytes.fromhex(tx['public_key']), curve=ecdsa.SECP256k1)
            
            try:
                # Use sigdecode_string for verification to match sigencode_string used in signing
                public_key.verify(signature, tx_bytes, hashfunc=hashlib.sha256, sigdecode=ecdsa.util.sigdecode_string)
                return True
            except ecdsa.BadSignatureError:
                logging.error(f"Invalid signature for transaction: {tx['txid'][:8]}")
                return False
                
        except Exception as e:
            logging.error(f"Error verifying transaction signature: {e}")
            return False

    def get_last_nonce(self, sender_address):
        """Get the last used nonce for a sender address"""
        # Check transaction pool first
        last_nonce = 0
        for tx in self.transaction_pool:
            if tx["sender_address"] == sender_address:
                last_nonce = max(last_nonce, tx["nonce"])

        # Check blockchain
        for block in self.chain:
            for tx in block.transactions:
                if tx["sender_address"] == sender_address:
                    last_nonce = max(last_nonce, tx["nonce"])

        return last_nonce

    def calculate_next_difficulty(self, actual_mining_time=None):
        self.difficulty_bits = 16.00
        target = int(self.calculate_target(self.difficulty_bits))  # Convert to int
        logging.info(f"🔧 Static Difficulty Set: Bits: {self.difficulty_bits:.2f}, Target: 0x{target:x}")
        return target

    def save_to_db(self):
        """Save blockchain state to disk"""
        try:
            if not self.db_file:
                return
                
            # Save chain metadata
            metadata = {
                "chain": [block.hash for block in self.chain],
                "utxo_set": self.utxo_set.to_dict(),
                "used_txids": list(self.used_txids),
                "transaction_pool": self.transaction_pool,
                "last_prune_height": self.last_prune_height,
                "header_archive": self.header_archive,
                "last_archive_height": self.last_archive_height
            }
            
            # Log UTXO set state before saving
            logging.info(f"Saving UTXO set with {len(self.utxo_set)} UTXOs")
            
            # Save to disk using SQLiteStorage
            self.storage.save_chain_metadata("chain_metadata", metadata)
            
            # Save blocks using BlockStorage
            for block in self.chain:
                self.block_storage.save_block(block)
                
            logging.info("Successfully saved blockchain state to disk")
                
        except Exception as e:
            logging.error(f"Error saving blockchain to disk: {e}")
            raise

    def load_from_db(self):
        """Load blockchain state from disk"""
        try:
            if not self.db_file:
                return
                
            # Load chain metadata using SQLiteStorage
            metadata = self.storage.load_chain_metadata("chain_metadata")
            if not metadata:
                return
                
            # Load UTXO set
            if "utxo_set" in metadata:
                try:
                    self.utxo_set = UTXOSet.from_dict(metadata["utxo_set"])
                    logging.info(f"Loaded UTXO set with {len(self.utxo_set)} UTXOs")
                except Exception as e:
                    logging.error(f"Error loading UTXO set: {e}")
                    # If UTXO set loading fails, rebuild it from blocks
                    self.utxo_set = UTXOSet()
                    logging.info("Rebuilding UTXO set from blocks")
                    for block in self.chain:
                        self.update_utxo_set(block)
                
            # Load used transaction IDs
            if "used_txids" in metadata:
                self.used_txids = set(metadata["used_txids"])
                logging.info(f"Loaded {len(self.used_txids)} used transaction IDs")
                
            # Load transaction pool
            if "transaction_pool" in metadata:
                self.transaction_pool = metadata["transaction_pool"]
                logging.info(f"Loaded {len(self.transaction_pool)} transactions from pool")
                
            # Load last prune height
            if "last_prune_height" in metadata:
                self.last_prune_height = metadata["last_prune_height"]
                
            # Load header archive
            if "header_archive" in metadata:
                self.header_archive = metadata["header_archive"]
                logging.info(f"Loaded {len(self.header_archive)} archived block headers")
                
            # Load last archive height
            if "last_archive_height" in metadata:
                self.last_archive_height = metadata["last_archive_height"]
                
            # Load blocks using BlockStorage
            for block_hash in metadata["chain"]:
                block = self.block_storage.load_block(block_hash)
                if block:
                    self.chain.append(block)
                    
            logging.info(f"Successfully loaded {len(self.chain)} blocks from disk")
                
        except Exception as e:
            logging.error(f"Error loading blockchain from disk: {e}")
            raise

    def _rebuild_chain_from_last_valid(self, last_valid_block):
        """Rebuild the chain from the last valid block"""
        try:
            logging.info(f"Starting chain rebuild from block {last_valid_block.index}")
            
            # Clear current chain and UTXO set
            self.chain = []
            self.utxo_set = UTXOSet()
            self.used_txids = set()
            
            # Add the last valid block
            self.chain.append(last_valid_block)
            current_height = len(self.chain)
            
            # Process the last valid block's transactions
            for tx in last_valid_block.transactions:
                if tx.is_coinbase():
                    # Add coinbase UTXO
                    utxo_id = f"{tx.txid}:0"
                    self.utxo_set.add_utxo(utxo_id, tx.outputs[0])
                    logging.info(f"Added coinbase UTXO {utxo_id} during rebuild")
                else:
                    # Add regular transaction outputs
                    for i, output in enumerate(tx.outputs):
                        utxo_id = f"{tx.txid}:{i}"
                        self.utxo_set.add_utxo(utxo_id, output)
                        logging.info(f"Added UTXO {utxo_id} during rebuild")
                    
                    # Mark inputs as spent
                    for input in tx.inputs:
                        utxo_id = f"{input.txid}:{input.vout}"
                        self.utxo_set.spend_utxo(utxo_id, current_height)
                        logging.info(f"Spent UTXO {utxo_id} during rebuild")
            
            # Load and process subsequent blocks
            current_block = last_valid_block
            while True:
                next_block = self.block_storage.load_block(current_block.next_hash)
                if not next_block:
                    break
                    
                if not self.validate_block(next_block):
                    logging.error(f"Invalid block {next_block.index} found during rebuild")
                    break
                    
                self.chain.append(next_block)
                current_block = next_block
                current_height = len(self.chain)
                
                # Process block transactions
                for tx in next_block.transactions:
                    if tx.is_coinbase():
                        # Add coinbase UTXO
                        utxo_id = f"{tx.txid}:0"
                        self.utxo_set.add_utxo(utxo_id, tx.outputs[0])
                        logging.info(f"Added coinbase UTXO {utxo_id} during rebuild")
                    else:
                        # Add regular transaction outputs
                        for i, output in enumerate(tx.outputs):
                            utxo_id = f"{tx.txid}:{i}"
                            self.utxo_set.add_utxo(utxo_id, output)
                            logging.info(f"Added UTXO {utxo_id} during rebuild")
                        
                        # Mark inputs as spent
                        for input in tx.inputs:
                            utxo_id = f"{input.txid}:{input.vout}"
                            self.utxo_set.spend_utxo(utxo_id, current_height)
                            logging.info(f"Spent UTXO {utxo_id} during rebuild")
            
            # Save the rebuilt state
            self.save_to_db()
            logging.info(f"Successfully rebuilt chain with {len(self.chain)} blocks")
            
        except Exception as e:
            logging.error(f"Error during chain rebuild: {e}")
            raise

    def to_dict(self):
        """Convert blockchain to dictionary format"""
        try:
            return {
                "chain": [block.hash for block in self.chain],
                "utxo_set": self.utxo_set.to_dict(),
                "used_txids": list(self.used_txids),
                "transaction_pool": self.transaction_pool
            }
        except Exception as e:
            logging.error(f"Error converting blockchain to dictionary: {e}")
            return {}

    def get_genesis_timestamp(self):
        """Get the timestamp of the genesis block"""
        try:
            return self.chain[0].timestamp if self.chain else float('inf')
        except Exception as e:
            logging.error(f"Error getting genesis timestamp: {e}")
            return float('inf')

    def validate_chain(self, chain):
        """Validate a chain of blocks"""
        try:
            # Check if chain is empty
            if not chain:
                logging.error("Chain is empty")
                return False
                
            # Check genesis block
            if chain[0].previous_hash != "0" * 64:
                logging.error("Invalid genesis block")
                return False
                
            # Find the closest UTXO set snapshot
            current_height = len(chain)
            if not self.utxo_set.restore_from_snapshot(current_height):
                logging.warning("No suitable UTXO set snapshot found, validating from genesis")
                
            # Check each block
            for i in range(1, len(chain)):
                current_block = chain[i]
                previous_block = chain[i-1]
                
                # Check block format
                if not self.validate_block_format(current_block):
                    logging.error(f"Invalid block format at height {i}")
                    return False
                    
                # Check block hash
                if not self.verify_block_hash(current_block):
                    logging.error(f"Invalid block hash at height {i}")
                    return False
                    
                # Check block signature
                if not self.verify_block_signature(current_block):
                    logging.error(f"Invalid block signature at height {i}")
                    return False
                    
                # Check block difficulty
                if not self.verify_block_difficulty(current_block):
                    logging.error(f"Block difficulty not met at height {i}")
                    return False
                    
                # Check previous hash
                if current_block.previous_hash != previous_block.hash:
                    logging.error(f"Invalid previous hash at height {i}")
                    return False
                    
                # Check timestamp
                if current_block.timestamp <= previous_block.timestamp:
                    logging.error(f"Invalid timestamp at height {i}")
                    return False
                    
                # Check transactions
                for tx in current_block.transactions:
                    if not self.validate_transaction_format(tx):
                        logging.error(f"Invalid transaction format in block {i}")
                        return False
                        
                    if not self.verify_transaction_signature(tx):
                        logging.error(f"Invalid transaction signature in block {i}")
                        return False
                        
                    if not self.verify_transaction_inputs(tx):
                        logging.error(f"Invalid transaction inputs in block {i}")
                        return False
                        
            return True
        except Exception as e:
            logging.error(f"Error validating chain: {e}")
            return False

    def verify_transaction_inputs(self, tx):
        """Verify that a transaction's inputs are valid and unspent"""
        try:
            # Skip verification for coinbase and genesis transactions
            if tx['sender_address'] in ["COINBASE", "GENESIS"]:
                return True
                
            # Check if inputs exist
            if not tx.get('inputs'):
                logging.error("Transaction has no inputs")
                return False
                
            # Check each input
            total_input_amount = 0
            for input_tx in tx['inputs']:
                # Create UTXO ID
                utxo_id = f"{input_tx['txid']}:{input_tx['output_index']}"
                
                # Check if input exists in UTXO set
                utxo = self.utxo_set.get_utxo(utxo_id)
                if not utxo:
                    logging.error(f"Input {utxo_id} not found in UTXO set")
                    return False
                    
                # Check if input is already spent
                if utxo.spent:
                    logging.debug(f"Input {utxo_id} already spent")
                    return False
                    
                # Check if input amount matches UTXO amount
                if float(input_tx['amount']) != float(utxo.amount):
                    logging.error(f"Input amount {input_tx['amount']} does not match UTXO amount {utxo.amount}")
                    return False
                    
                # Check if input address matches UTXO address
                if input_tx['address'] != utxo.address:
                    logging.error(f"Input address {input_tx['address']} does not match UTXO address {utxo.address}")
                    return False
                    
                total_input_amount += float(input_tx['amount'])
            
            # Check if total input amount covers outputs plus fee
            total_output_amount = sum(float(output['amount']) for output in tx['outputs'])
            if total_input_amount < total_output_amount + float(tx['fee']):
                logging.error(f"Total input amount {total_input_amount} less than outputs {total_output_amount} plus fee {tx['fee']}")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Error verifying transaction inputs: {e}")
            return False

    def get_transaction_pool(self):
        """Get the transaction pool"""
        try:
            return self.transaction_pool
        except Exception as e:
            logging.error(f"Error getting transaction pool: {e}")
            return []

    def get_pool_transactions(self):
        """Get transactions from the transaction pool"""
        try:
            return self.transaction_pool
        except Exception as e:
            logging.error(f"Error getting transactions from pool: {e}")
            return []

    def get_chain(self):
        """Get the blockchain"""
        try:
            return self.chain
        except Exception as e:
            logging.error(f"Error getting chain: {e}")
            return []

    def get_block(self, block_hash):
        """Get a block by hash"""
        try:
            for block in self.chain:
                if block.hash == block_hash:
                    return block
            return None
        except Exception as e:
            logging.error(f"Error getting block {block_hash[:8]}: {e}")
            return None

    def get_transaction(self, txid):
        """Get a transaction by ID"""
        try:
            for block in self.chain:
                for tx in block.transactions:
                    if tx["txid"] == txid:
                        return tx
            return None
        except Exception as e:
            logging.error(f"Error getting transaction {txid[:8]}: {e}")
            return None

    def get_blocks_by_address(self, address):
        """Get blocks by address"""
        try:
            blocks = []
            for block in self.chain:
                for tx in block.transactions:
                    # Check inputs
                    for input_tx in tx["inputs"]:
                        if input_tx["address"] == address:
                            blocks.append(block)
                            break
                    # Check outputs
                    for output in tx["outputs"]:
                        if output["address"] == address:
                            blocks.append(block)
                            break
            return blocks
        except Exception as e:
            logging.error(f"Error getting blocks for {address}: {e}")
            return []

    def process_block(self, block):
        """Process a new block"""
        try:
            # Check if block has already been processed
            with self.chain_lock:
                if any(b.hash == block.hash for b in self.chain):
                    logging.debug(f"Block {block.hash[:8]} already in chain, processing transactions")
                    # Process transactions even if block is already in chain
                    current_height = len(self.chain)
                    for tx in block.transactions:
                        # Convert to dictionary if it's a Transaction object
                        tx_dict = tx.to_dict() if hasattr(tx, 'to_dict') else tx
                        
                        # Skip if transaction already processed
                        if tx_dict['txid'] in self.used_txids:
                            logging.debug(f"Transaction {tx_dict['txid'][:8]} already processed, skipping")
                            continue
                        
                        # Handle genesis transaction
                        if tx_dict.get('sender_address') == 'GENESIS':
                            for i, output in enumerate(tx_dict['outputs']):
                                utxo_id = f"{tx_dict['txid']}:{i}"
                                existing_utxo = self.utxo_set.get_utxo(utxo_id)
                                if not existing_utxo:
                                    utxo = UTXO(tx_dict['txid'], i, output['amount'], output['address'])
                                    self.utxo_set.add_utxo(utxo)
                                    logging.info(f"Added genesis UTXO {utxo.get_id()} with amount {output['amount']} for {output['address']}")
                            self.used_txids.add(tx_dict['txid'])
                            continue
                            
                        # Handle coinbase transaction
                        if tx_dict.get('sender_address') == 'COINBASE':
                            for i, output in enumerate(tx_dict['outputs']):
                                utxo_id = f"{tx_dict['txid']}:{i}"
                                existing_utxo = self.utxo_set.get_utxo(utxo_id)
                                if not existing_utxo:
                                    utxo = UTXO(tx_dict['txid'], i, output['amount'], output['address'])
                                    self.utxo_set.add_utxo(utxo)
                                    logging.debug(f"Added coinbase UTXO {utxo.get_id()} with amount {output['amount']} for {output['address']}")
                            self.used_txids.add(tx_dict['txid'])
                            continue
                            
                        # Handle regular transactions
                        # First check all inputs exist and are unspent
                        for input_tx in tx_dict['inputs']:
                            utxo_id = f"{input_tx['txid']}:{input_tx['output_index']}"
                            utxo = self.utxo_set.get_utxo(utxo_id)
                            if not utxo or utxo.spent:
                                logging.error(f"Input UTXO {utxo_id} not found or already spent")
                                return False
                                
                        # Mark inputs as spent
                        for input_tx in tx_dict['inputs']:
                            utxo_id = f"{input_tx['txid']}:{input_tx['output_index']}"
                            current_height = len(self.chain)
                            self.utxo_set.spend_utxo(utxo_id, current_height)
                            logging.info(f"Spent UTXO {utxo_id} for transaction {tx_dict['txid'][:8]}")
                            
                        # Add new outputs
                        for i, output in enumerate(tx_dict['outputs']):
                            utxo_id = f"{tx_dict['txid']}:{i}"
                            existing_utxo = self.utxo_set.get_utxo(utxo_id)
                            if not existing_utxo:
                                utxo = UTXO(tx_dict['txid'], i, output['amount'], output['address'])
                                self.utxo_set.add_utxo(utxo)
                                logging.info(f"Added UTXO {utxo.get_id()} with amount {output['amount']} for {output['address']}")
                                
                        self.used_txids.add(tx_dict['txid'])
                    
                    # Update balances
                    if not self.update_balances(block):
                        logging.debug(f"Failed to update balances for block {block.hash[:8]}")
                        return False
                    
                    # Save to disk
                    self.save_to_db()
                    
                    return True
                
                # Validate block before processing
                if not self.validate_block(block):
                    logging.error(f"Invalid block {block.hash[:8]}")
                    return False
                
                # Add block to chain
                self.chain.append(block)
                current_height = len(self.chain)
                logging.info(f"Added block {block.hash[:8]} to chain at height {current_height-1}")
                
                # Process transactions
                for tx in block.transactions:
                    # Convert to dictionary if it's a Transaction object
                    tx_dict = tx.to_dict() if hasattr(tx, 'to_dict') else tx
                    
                    # Skip if transaction already processed
                    if tx_dict['txid'] in self.used_txids:
                        logging.debug(f"Transaction {tx_dict['txid'][:8]} already processed, skipping")
                        continue
                    
                    # Handle genesis transaction
                    if tx_dict.get('sender_address') == 'GENESIS':
                        for i, output in enumerate(tx_dict['outputs']):
                            utxo_id = f"{tx_dict['txid']}:{i}"
                            existing_utxo = self.utxo_set.get_utxo(utxo_id)
                            if not existing_utxo:
                                utxo = UTXO(tx_dict['txid'], i, output['amount'], output['address'])
                                self.utxo_set.add_utxo(utxo)
                                logging.info(f"Added genesis UTXO {utxo.get_id()} with amount {output['amount']} for {output['address']}")
                        self.used_txids.add(tx_dict['txid'])
                        continue
                        
                    # Handle coinbase transaction
                    if tx_dict.get('sender_address') == 'COINBASE':
                        for i, output in enumerate(tx_dict['outputs']):
                            utxo_id = f"{tx_dict['txid']}:{i}"
                            existing_utxo = self.utxo_set.get_utxo(utxo_id)
                            if not existing_utxo:
                                utxo = UTXO(tx_dict['txid'], i, output['amount'], output['address'])
                                self.utxo_set.add_utxo(utxo)
                                logging.debug(f"Added coinbase UTXO {utxo.get_id()} with amount {output['amount']} for {output['address']}")
                        self.used_txids.add(tx_dict['txid'])
                        continue
                        
                    # Handle regular transactions
                    # First check all inputs exist and are unspent
                    for input_tx in tx_dict['inputs']:
                        utxo_id = f"{input_tx['txid']}:{input_tx['output_index']}"
                        utxo = self.utxo_set.get_utxo(utxo_id)
                        if not utxo or utxo.spent:
                            logging.error(f"Input UTXO {utxo_id} not found or already spent")
                            return False
                            
                    # Mark inputs as spent
                    for input_tx in tx_dict['inputs']:
                        utxo_id = f"{input_tx['txid']}:{input_tx['output_index']}"
                        current_height = len(self.chain)
                        self.utxo_set.spend_utxo(utxo_id, current_height)
                        logging.info(f"Spent UTXO {utxo_id} for transaction {tx_dict['txid'][:8]}")
                            
                    # Add new outputs
                    for i, output in enumerate(tx_dict['outputs']):
                        utxo_id = f"{tx_dict['txid']}:{i}"
                        existing_utxo = self.utxo_set.get_utxo(utxo_id)
                        if not existing_utxo:
                            utxo = UTXO(tx_dict['txid'], i, output['amount'], output['address'])
                            self.utxo_set.add_utxo(utxo)
                            logging.info(f"Added UTXO {utxo.get_id()} with amount {output['amount']} for {output['address']}")
                            
                    self.used_txids.add(tx_dict['txid'])
                
                # Update balances
                if not self.update_balances(block):
                    logging.error(f"Failed to update balances for block {block.hash[:8]}")
                    return False
                
                # Save to disk
                self.save_to_db()
                
                return True
        except Exception as e:
            logging.error(f"Error processing block: {e}")
            return False

    def get_height(self):
        """Get the current height of the blockchain."""
        return len(self.chain)

    def add_orphan_block(self, block):
        """Add an orphan block to storage with proper validation"""
        try:
            # Validate block format
            if not self.validate_block_format(block):
                logging.error("Invalid orphan block format")
                return False
                
            # Check if block is already known
            if block.hash in self.orphan_blocks or block.hash in [b.hash for b in self.chain]:
                logging.warning("Block already known")
                return False
                
            # Verify block signature
            if not self.verify_block_signature(block):
                logging.error("Invalid orphan block signature")
                return False
                
            # Verify block hash
            if not self.verify_block_hash(block):
                logging.error("Invalid orphan block hash")
                return False
                
            # Store orphan block
            self.orphan_blocks[block.hash] = block
            self.orphan_timestamps[block.hash] = time.time()
            
            # Track dependencies
            self.orphan_dependencies[block.hash] = block.previous_hash
            
            # Cleanup old orphan blocks
            self.cleanup_orphan_blocks()
            
            return True
        except Exception as e:
            logging.error(f"Error adding orphan block: {e}")
            return False

    def cleanup_orphan_blocks(self):
        """Cleanup old orphan blocks"""
        try:
            current_time = time.time()
            expired_blocks = []
            
            # Find expired blocks
            for block_hash, timestamp in self.orphan_timestamps.items():
                if current_time - timestamp > self.MAX_ORPHAN_AGE:
                    expired_blocks.append(block_hash)
                    
            # Remove expired blocks
            for block_hash in expired_blocks:
                del self.orphan_blocks[block_hash]
                del self.orphan_dependencies[block_hash]
                del self.orphan_timestamps[block_hash]
                
            # Remove excess blocks if needed
            while len(self.orphan_blocks) > self.MAX_ORPHAN_COUNT:
                oldest_block = min(self.orphan_timestamps.items(), key=lambda x: x[1])[0]
                del self.orphan_blocks[oldest_block]
                del self.orphan_dependencies[oldest_block]
                del self.orphan_timestamps[oldest_block]
        except Exception as e:
            logging.error(f"Error cleaning up orphan blocks: {e}")

    def process_orphan_blocks(self):
        """Process orphan blocks and handle chain forks"""
        try:
            # Check for forks periodically
            current_time = time.time()
            if current_time - self.last_fork_check >= self.FORK_CHECK_INTERVAL:
                self.check_for_forks()
                self.last_fork_check = current_time
                
            # Process orphan blocks
            processed_blocks = set()
            while True:
                found_new = False
                
                for block_hash, block in list(self.orphan_blocks.items()):
                    if block_hash in processed_blocks:
                        continue
                        
                    # Check if parent exists
                    parent_hash = self.orphan_dependencies[block_hash]
                    if parent_hash in [b.hash for b in self.chain]:
                        # Add block to chain
                        logging.info(f"Processing orphan block {block_hash[:8]} with parent {parent_hash[:8]}")
                        if self.add_block(block):
                            processed_blocks.add(block_hash)
                            found_new = True
                            logging.info(f"Successfully processed orphan block {block_hash[:8]}")
                        else:
                            logging.error(f"Failed to process orphan block {block_hash[:8]}")
                            
                if not found_new:
                    break
                    
            # Cleanup processed blocks
            for block_hash in processed_blocks:
                if block_hash in self.orphan_blocks:
                    del self.orphan_blocks[block_hash]
                    del self.orphan_dependencies[block_hash]
                    del self.orphan_timestamps[block_hash]
                    
        except Exception as e:
            logging.error(f"Error processing orphan blocks: {e}")

    def check_for_forks(self):
        """Check for chain forks and handle them"""
        try:
            # Get current chain tip
            current_tip = self.chain[-1].hash if self.chain else None
            
            # Check for competing chains
            for block_hash, block in self.orphan_blocks.items():
                # Build chain from orphan block
                chain = self.build_chain_from_block(block)
                if not chain:
                    continue
                    
                # Store competing chain
                self.fork_chains[block_hash] = chain
                
            # Resolve forks if needed
            if len(self.fork_chains) > 0:
                self.resolve_forks()
                
        except Exception as e:
            logging.error(f"Error checking for forks: {e}")

    def build_chain_from_block(self, block):
        """Build chain from a block"""
        try:
            chain = [block]
            current_hash = block.previous_hash
            
            # Build chain backwards
            while current_hash:
                # Check main chain
                for b in reversed(self.chain):
                    if b.hash == current_hash:
                        chain.append(b)
                        return chain
                        
                # Check orphan blocks
                if current_hash in self.orphan_blocks:
                    chain.append(self.orphan_blocks[current_hash])
                    current_hash = self.orphan_blocks[current_hash].previous_hash
                else:
                    break
                    
            return None
        except Exception as e:
            logging.error(f"Error building chain from block: {e}")
            return None

    def resolve_forks(self):
        """Resolve competing chains"""
        try:
            # Get current chain length
            current_length = len(self.chain)
            
            # Find longest valid chain
            longest_chain = None
            longest_length = current_length
            
            for chain in self.fork_chains.values():
                # Validate chain
                if not self.validate_chain(chain):
                    continue
                    
                # Check if longer than current chain
                if len(chain) > longest_length:
                    longest_chain = chain
                    longest_length = len(chain)
                    
            # Reorganize to longest chain if found
            if longest_chain:
                if self._reorganize_chain(longest_chain):
                    logging.info(f"Reorganized to longer chain with length {longest_length}")
                else:
                    logging.error("Failed to reorganize to longer chain")
                    
            # Clear fork chains
            self.fork_chains.clear()
            
        except Exception as e:
            logging.error(f"Error resolving forks: {e}")

    def validate_chain(self, chain):
        """Validate a chain of blocks"""
        try:
            # Check each block
            for i in range(len(chain)):
                block = chain[i]
                
                # Validate block
                if not self.validate_block(block):
                    return False
                    
                # Check previous hash
                if i > 0 and block.previous_hash != chain[i-1].hash:
                    return False
                    
            return True
        except Exception as e:
            logging.error(f"Error validating chain: {e}")
            return False

    def get_block_size_limit(self):
        """Get current block size limit"""
        return self.MAX_BLOCK_SIZE

    def get_block_size_history(self):
        """Get block size history"""
        return self.block_size_history

    def get_average_block_size(self):
        """Get average block size"""
        if not self.block_size_history:
            return 0
        return sum(self.block_size_history) / len(self.block_size_history)

    def update_utxo_set(self, block):
        """Update UTXO set with transactions from a block"""
        try:
            for tx in block.transactions:
                # Convert to dictionary if it's a Transaction object
                tx_dict = tx.to_dict() if hasattr(tx, 'to_dict') else tx
                
                # Skip if transaction already processed
                if tx_dict['txid'] in self.used_txids:
                    logging.debug(f"Transaction {tx_dict['txid'][:8]} already processed, skipping")
                    continue
                
                # Handle genesis transaction
                if tx_dict.get('sender_address') == 'GENESIS':
                    for i, output in enumerate(tx_dict['outputs']):
                        utxo_id = f"{tx_dict['txid']}:{i}"
                        existing_utxo = self.utxo_set.get_utxo(utxo_id)
                        if not existing_utxo:
                            utxo = UTXO(tx_dict['txid'], i, output['amount'], output['address'])
                            self.utxo_set.add_utxo(utxo)
                            logging.info(f"Added genesis UTXO {utxo.get_id()} with amount {output['amount']} for {output['address']}")
                    self.used_txids.add(tx_dict['txid'])
                    continue
                    
                # Handle coinbase transaction
                if tx_dict.get('sender_address') == 'COINBASE':
                    for i, output in enumerate(tx_dict['outputs']):
                        utxo_id = f"{tx_dict['txid']}:{i}"
                        existing_utxo = self.utxo_set.get_utxo(utxo_id)
                        if not existing_utxo:
                            utxo = UTXO(tx_dict['txid'], i, output['amount'], output['address'])
                            self.utxo_set.add_utxo(utxo)
                            logging.debug(f"Added coinbase UTXO {utxo.get_id()} with amount {output['amount']} for {output['address']}")
                    self.used_txids.add(tx_dict['txid'])
                    continue
                    
                # Handle regular transactions
                # First check all inputs exist and are unspent
                for input_tx in tx_dict['inputs']:
                    utxo_id = f"{input_tx['txid']}:{input_tx['output_index']}"
                    utxo = self.utxo_set.get_utxo(utxo_id)
                    if not utxo or utxo.spent:
                        logging.error(f"Input UTXO {utxo_id} not found or already spent")
                        return False
                        
                # Mark inputs as spent
                for input_tx in tx_dict['inputs']:
                    utxo_id = f"{input_tx['txid']}:{input_tx['output_index']}"
                    current_height = len(self.chain)
                    self.utxo_set.spend_utxo(utxo_id, current_height)
                    logging.info(f"Spent UTXO {utxo_id} for transaction {tx_dict['txid'][:8]}")
                        
                # Add new outputs
                for i, output in enumerate(tx_dict['outputs']):
                    utxo_id = f"{tx_dict['txid']}:{i}"
                    existing_utxo = self.utxo_set.get_utxo(utxo_id)
                    if not existing_utxo:
                        utxo = UTXO(tx_dict['txid'], i, output['amount'], output['address'])
                        self.utxo_set.add_utxo(utxo)
                        logging.info(f"Added UTXO {utxo.get_id()} with amount {output['amount']} for {output['address']}")
                        
                self.used_txids.add(tx_dict['txid'])
            
            return True
        except Exception as e:
            logging.error(f"Error updating UTXO set: {e}")
            return False

    def get_transactions(self, bloom_filter_b64, start_block=0, end_block=None):
        """Get transactions matching the bloom filter"""
        try:
            # Create bloom filter from base64 with same parameters as light wallet
            bloom_filter = BloomFilter.from_base64(bloom_filter_b64, capacity=10000, error_rate=0.001)
            if not bloom_filter:
                logging.error("Failed to create bloom filter from base64")
                return []
            
            # Get block range
            if end_block is None:
                end_block = len(self.chain)
            else:
                end_block = min(end_block, len(self.chain))
            
            # Process all blocks
            transactions = []
            for block in self.chain[start_block:end_block]:
                for tx_dict in block.transactions:
                    # Special handling for genesis transactions
                    is_genesis = (tx_dict.get("sender_address") == "GENESIS" or tx_dict.get("txid") == "0" * 64)
                    if is_genesis:
                        # For genesis transactions, always include if GENESIS is in filter
                        if bloom_filter.contains("GENESIS".encode()) or any(bloom_filter.contains(output["address"].encode()) for output in tx_dict.get("outputs", [])):
                            transactions.append(tx_dict)
                            continue
                    
                    # For regular transactions, check inputs and outputs
                    should_add = False
                    
                    # Check sender address
                    if tx_dict.get("sender_address") and bloom_filter.contains(tx_dict["sender_address"].encode()):
                        should_add = True
                    
                    # Check output addresses
                    if not should_add:
                        for output in tx_dict.get("outputs", []):
                            if bloom_filter.contains(output["address"].encode()):
                                should_add = True
                                break
                    
                    if should_add:
                        transactions.append(tx_dict)
            
            return transactions
            
        except Exception as e:
            logging.error(f"Error getting transactions: {e}")
            return []

    def add_transaction(self, transaction):
        """Add a transaction to the pool if valid"""
        try:
            # Convert transaction to dict if needed
            tx_dict = transaction.to_dict() if hasattr(transaction, 'to_dict') else transaction
            
            # Check if transaction is already in pool or used
            if tx_dict["txid"] in self.used_txids:
                logging.debug(f"Transaction {tx_dict['txid']} already used")
                return {"type": "error", "error": "Transaction already used"}

            # Check if transaction is already in pool
            for tx in self.transaction_pool:
                if tx["txid"] == tx_dict["txid"]:
                    logging.debug(f"Transaction {tx_dict['txid']} already in pool")
                    return {"type": "error", "error": "Transaction already in pool"}

            # Validate transaction format
            if not self.validate_transaction_format(tx_dict):
                logging.debug(f"Invalid transaction format for {tx_dict['txid']}")
                return {"type": "error", "error": "Invalid transaction format"}

            # Add to pool and sort by fee
            self.transaction_pool.append(tx_dict)
            self.sort_transaction_pool()
            logging.debug(f"Added transaction {tx_dict['txid']} to pool")
            return {"type": "success", "txid": tx_dict["txid"]}
            
        except Exception as e:
            logging.error(f"Error adding transaction: {e}")
            return {"type": "error", "error": str(e)}

    def get_utxos(self, address):
        """Get all UTXOs for an address"""
        try:
            # Get all UTXOs for the address
            utxos = self.utxo_set.get_address_utxos(address)
            
            # Convert UTXOs to dictionary format
            utxo_list = []
            for utxo in utxos:
                if not utxo.spent:
                    utxo_list.append({
                        "txid": utxo.txid,
                        "index": utxo.index,
                        "amount": float(utxo.amount),
                        "address": utxo.address
                    })
                    
            logging.info(f"Found {len(utxo_list)} UTXOs for {address}")
            return utxo_list
        except Exception as e:
            logging.error(f"Error getting UTXOs: {e}")
            return []

    def get_transaction_history(self, address):
        """Get transaction history for an address"""
        try:
            transactions = []
            seen_txids = set()
            for block in self.chain:
                for tx in block.transactions:
                    # Skip if we've already seen this transaction
                    txid = tx.get('txid')
                    if not txid or txid in seen_txids:
                        continue
                        
                    # Check if address is involved in the transaction
                    is_sender = tx.get('sender_address') == address
                    is_receiver = any(output.get('address') == address for output in tx.get('outputs', []))
                    
                    if is_sender or is_receiver:
                        transactions.append(tx)
                        seen_txids.add(txid)
                        
            return sorted(transactions, key=lambda x: x.get('timestamp', 0))
        except Exception as e:
            logging.error(f"Error getting transaction history: {e}")
            return []

    def mine_block(self, transactions=None):
        """Mine a new block"""
        try:
            # Get transactions for the block
            if transactions is None:
                transactions = self.get_transactions_for_block()
                
            # Create new block
            previous_hash = self.chain[-1].hash if self.chain else "0" * 64
            index = len(self.chain)
            
            # Create block
            block = Block(
                version=1,
                previous_hash=previous_hash,
                transactions=transactions,
                difficulty_target=self.difficulty_target,
                nonce=0,
                timestamp=int(time.time()),
                index=index
            )
            
            # Mine block
            logging.info(f"Mining block {index} with {len(transactions)} transactions")
            while True:
                # Check if block has already been mined
                if any(b.hash == block.hash for b in self.chain):
                    logging.info(f"Block {block.hash[:8]} already mined, skipping")
                    return None
                    
                # Try to find valid nonce
                if self.check_proof_of_work(block.to_dict()):
                    logging.debug(f"Mined block {index} with hash {block.hash}")
                    return block
                    
                # Increment nonce
                block.nonce += 1
                
        except Exception as e:
            logging.error(f"Error mining block: {e}")
            return None

    def get_utxo_snapshot_info(self):
        """Get information about UTXO set snapshots"""
        return self.utxo_set.get_snapshot_info()

    def prune_blocks(self):
        """Prune old blocks, keeping only recent ones"""
        try:
            current_height = len(self.chain)
            
            # Check if we should prune
            if current_height - self.last_prune_height < self.pruning_interval:
                return
                
            # Calculate blocks to keep
            keep_height = max(0, current_height - self.KEEP_RECENT_BLOCKS)
            
            # Get blocks to prune
            blocks_to_prune = self.chain[:keep_height]
            
            # Remove blocks from storage
            for block in blocks_to_prune:
                self.block_storage.delete_block(block.hash)
                
            # Update chain
            self.chain = self.chain[keep_height:]
            
            # Update last prune height
            self.last_prune_height = current_height
            
            # Log pruning operation
            logging.info(f"Pruned {len(blocks_to_prune)} blocks, keeping {len(self.chain)} recent blocks")
            
        except Exception as e:
            logging.error(f"Error pruning blocks: {e}")

    def archive_blocks(self):
        """Archive old blocks, keeping only headers"""
        try:
            current_height = len(self.chain)
            
            # Check if we should archive
            if current_height - self.last_archive_height < self.archive_interval:
                return
                
            # Calculate blocks to archive
            keep_height = max(0, current_height - self.KEEP_FULL_BLOCKS)
            
            # Get blocks to archive
            blocks_to_archive = self.chain[:keep_height]
            
            # Archive blocks
            for block in blocks_to_archive:
                # Store header only
                header = {
                    'version': block.version,
                    'previous_hash': block.previous_hash,
                    'merkle_root': block.merkle_root,
                    'timestamp': block.timestamp,
                    'difficulty_target': block.difficulty_target,
                    'nonce': block.nonce,
                    'hash': block.hash,
                    'index': block.index
                }
                self.header_archive[block.hash] = header
                
                # Delete full block data
                self.block_storage.delete_block(block.hash)
                
            # Update chain to keep only recent blocks
            self.chain = self.chain[keep_height:]
            
            # Update last archive height
            self.last_archive_height = current_height
            
            # Log archiving operation
            logging.info(f"Archived {len(blocks_to_archive)} blocks, keeping {len(self.chain)} recent blocks")
            
        except Exception as e:
            logging.error(f"Error archiving blocks: {e}")

    def get_block_header(self, block_hash):
        """Get block header by hash"""
        try:
            # Check recent blocks first
            for block in self.chain:
                if block.hash == block_hash:
                    return {
                        'version': block.version,
                        'previous_hash': block.previous_hash,
                        'merkle_root': block.merkle_root,
                        'timestamp': block.timestamp,
                        'difficulty_target': block.difficulty_target,
                        'nonce': block.nonce,
                        'hash': block.hash,
                        'index': block.index
                    }
            
            # Check archived headers
            if block_hash in self.header_archive:
                return self.header_archive[block_hash]
                
            return None
        except Exception as e:
            logging.error(f"Error getting block header: {e}")
            return None

    def get_metrics(self):
        """Get blockchain metrics"""
        try:
            # Get current mining status
            mining_status = "idle"
            if hasattr(self, 'miner') and self.miner:
                # Check if there are transactions in the pool that could be mined
                if self.transaction_pool:
                    mining_status = "mining"
            
            # Get blockchain metrics
            chain_length = len(self.chain)
            last_block = self.chain[-1] if chain_length > 0 else None
            
            # Calculate network metrics
            total_peers = 0
            active_peers = 0
            if hasattr(self, 'p2p') and self.p2p:
                total_peers = len(self.p2p.peers)
                active_peers = sum(1 for peer in self.p2p.peers.values() if peer.is_connected())
            
            metrics = {
                "node": {
                    "status": "online",
                    "peers": total_peers,
                    "active_peers": active_peers
                },
                "blockchain": {
                    "chain_length": chain_length,
                    "last_block": {
                        "index": last_block.index if last_block else None,
                        "hash": last_block.hash if last_block else None
                    },
                    "mining": {
                        "status": mining_status,
                        "transaction_pool_size": len(self.transaction_pool)
                    }
                }
            }
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error getting metrics: {e}")
            return {
                "node": {
                    "status": "error",
                    "peers": 0,
                    "active_peers": 0
                },
                "blockchain": {
                    "chain_length": 0,
                    "last_block": {
                        "index": None,
                        "hash": None
                    },
                    "mining": {
                        "status": "error",
                        "transaction_pool_size": 0
                    }
                }
            }
