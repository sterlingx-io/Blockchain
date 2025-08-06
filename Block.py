import hashlib
import json
import time
import logging

class Block:
    def __init__(self, version=1, previous_hash=None, transactions=None, difficulty_target=16, nonce=0, timestamp=None, index=None):
        """Initialize a new block."""
        self.version = version
        self.previous_hash = previous_hash or "0" * 64
        self.transactions = []
        if transactions:
            for tx in transactions:
                if isinstance(tx, dict):
                    self.transactions.append(tx)
                else:
                    self.transactions.append(tx.to_dict())
        self.difficulty_target = difficulty_target
        self.nonce = nonce
        # Set timestamp based on whether this is a genesis block
        if self.previous_hash == "0" * 64:
            self.timestamp = 0  # Genesis block must have timestamp 0
        else:
            self.timestamp = timestamp or int(time.time())
        self.index = index
        self.merkle_root = self.calculate_merkle_root()
        self.hash = None
        self.update_hash()

    def calculate_hash(self):
        """Calculate the block hash."""
        block_header = {
            'version': self.version,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'timestamp': self.timestamp,
            'difficulty_target': self.difficulty_target,
            'nonce': self.nonce,
            'index': self.index
        }
        header_bytes = json.dumps(block_header, sort_keys=True).encode()
        return hashlib.sha256(header_bytes).hexdigest()

    def update_hash(self):
        """Update the block hash."""
        self.hash = self.calculate_hash()

    def calculate_merkle_root(self):
        """Calculate the Merkle root of the transactions."""
        if not self.transactions:
            return "0" * 64

        # Convert transactions to strings and hash them
        tx_hashes = []
        for tx in self.transactions:
            # Special handling for genesis transactions
            if isinstance(tx, dict) and tx.get("sender_address") == "GENESIS":
                tx_str = json.dumps({
                    "txid": tx["txid"],
                    "sender_address": tx["sender_address"],
                    "timestamp": tx["timestamp"],
                    "outputs": tx["outputs"]
                }, sort_keys=True)
            else:
                tx_str = json.dumps(tx, sort_keys=True)
            tx_hashes.append(hashlib.sha256(tx_str.encode()).hexdigest())

        # Build Merkle tree
        while len(tx_hashes) > 1:
            if len(tx_hashes) % 2 == 1:
                tx_hashes.append(tx_hashes[-1])
            tx_hashes = [hashlib.sha256((h1 + h2).encode()).hexdigest() for h1, h2 in zip(tx_hashes[::2], tx_hashes[1::2])]

        return tx_hashes[0]

    def to_dict(self):
        """Convert block to dictionary."""
        return {
            'version': self.version,
            'previous_hash': self.previous_hash,
            'transactions': self.transactions,  # Already in dict format
            'difficulty_target': self.difficulty_target,
            'nonce': self.nonce,
            'timestamp': self.timestamp,
            'hash': self.hash,
            'index': self.index,
            'merkle_root': self.merkle_root
        }

    def to_json(self):
        """Convert block to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data):
        """Create block from dictionary."""
        block = cls(
            version=data.get('version', 1),
            previous_hash=data.get('previous_hash'),
            transactions=data.get('transactions', []),
            difficulty_target=data.get('difficulty_target', 16),
            nonce=data.get('nonce', 0),
            timestamp=data.get('timestamp'),
            index=data.get('index')
        )
        block.hash = data.get('hash')
        block.merkle_root = data.get('merkle_root', block.calculate_merkle_root())
        return block

    def is_valid_proof(self):
        """Check if the block's hash meets the difficulty target"""
        try:
            # Ensure hash is calculated and is a valid string
            if not self.hash:
                self.update_hash()
            
            # Convert block hash to integer
            hash_int = int(self.hash, 16)
            
            # Calculate target (2^256 / (2^difficulty_target))
            target = 2**(256 - self.difficulty_target)
            
            # Check if hash is less than target
            return hash_int < target
        except Exception as e:
            logging.error(f"Error in is_valid_proof: {e}")
            return False
