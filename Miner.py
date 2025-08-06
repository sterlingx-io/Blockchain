import time
import heapq
from multiprocessing import Pool, cpu_count
from Block import Block
from Transaction import Transaction
import logging
import hashlib
import multiprocessing
import json

class Miner:
    def __init__(self, blockchain, wallet):
        self.blockchain = blockchain
        self.wallet = wallet
        self.BURN_ADDRESS = "0" * 32  # Define burn address
        self.INITIAL_REWARD = 50  # Initial mining reward
        self.HALVING_INTERVAL = 210000  # Number of blocks between halvings

    def calculate_mining_reward(self, block_height):
        """Calculate the mining reward based on block height."""
        halvings = block_height // self.HALVING_INTERVAL
        return self.INITIAL_REWARD / (2 ** halvings)

    def create_coinbase_transaction(self, block_height):
        """Create a coinbase transaction that burns the mining reward."""
        reward = self.calculate_mining_reward(block_height)
        timestamp = int(time.time())
        
        # Create transaction object
        tx = Transaction(
            inputs=[],  # Coinbase transactions have no inputs
            outputs=[
                {
                    "address": self.BURN_ADDRESS,
                    "amount": float(reward)  # Ensure amount is a float
                }
            ],
            fee=0,
            nonce=0,  # Coinbase transactions always have nonce 0
            timestamp=timestamp,
            sender_address="COINBASE",  # Use COINBASE as sender
            public_key="COINBASE",  # Set public key for coinbase
            signature="COINBASE" * 16,  # Set signature for coinbase
            network_id="mainnet",
            chain_id="1"
        )
        
        # Convert to dictionary and add type field
        tx_dict = tx.to_dict()
        tx_dict["type"] = "coinbase"
        return tx_dict

    def create_fee_burn_transaction(self, total_fees):
        """Create a fee burn transaction."""
        # Create fee burn transaction object
        fee_burn_tx = Transaction(
            inputs=[],
            outputs=[
                {
                    "address": self.BURN_ADDRESS,
                    "amount": float(total_fees)
                }
            ],
            fee=0,
            timestamp=int(time.time()),
            sender_address="FEES",
            public_key="FEES",  # Set public key for fee burn
            signature="FEES" * 16,  # Set signature for fee burn
            network_id="mainnet",
            chain_id="1"
        )
        # Convert to dictionary and add type field
        fee_burn_dict = fee_burn_tx.to_dict()
        fee_burn_dict["type"] = "feeburn"
        return fee_burn_dict

    def mine_block(self, transactions=None, block_height=None):
        """Mine a new block with the given transactions."""
        try:
            if transactions is None:
                transactions = []
            
            # Calculate total fees
            total_fees = sum(tx.get('fee', 0) for tx in transactions)
            
            # Create genesis block if chain is empty
            if len(self.blockchain.chain) == 0:
                # Create genesis transaction
                genesis_tx = Transaction(
                    outputs=[{
                        'address': self.wallet.address,  # Use actual wallet address
                        'amount': 60000000000,  # 60 billion coins
                        'index': 0
                    }],
                    inputs=[],
                    timestamp=0,  # Use 0 for genesis timestamp
                    sender_address="GENESIS",
                    public_key="GENESIS",
                    signature="0" * 128,  # Use 128 zeros for genesis signature
                    network_id="mainnet",
                    chain_id="1",
                    fee=0,
                    nonce=0
                )
                
                # Convert to dictionary and add type field
                tx_dict = genesis_tx.to_dict()
                tx_dict["type"] = "genesis"
                tx_dict["txid"] = "0" * 64  # Use 64 zeros for genesis txid
                
                # Create genesis block
                block = Block(
                    version=1,
                    previous_hash="0" * 64,  # Genesis block has no previous hash
                    transactions=[tx_dict],
                    difficulty_target=self.blockchain.difficulty_target,
                    nonce=0,
                    timestamp=0,  # Use 0 for genesis timestamp
                    index=0  # Genesis block has index 0
                )
            else:
                # Get the previous block to determine the next index
                previous_block = self.blockchain.chain[-1]
                next_index = previous_block.index + 1
                
                # Create coinbase transaction for regular blocks
                coinbase_tx = self.create_coinbase_transaction(next_index)
                
                # Create fee burning transaction if there are fees
                if total_fees > 0:
                    fee_burn_dict = self.create_fee_burn_transaction(total_fees)
                    # Add fee burn transaction after coinbase
                    block_transactions = [coinbase_tx, fee_burn_dict] + transactions
                    logging.info(f"Added coinbase and fee burn transactions to block")
                else:
                    # Add coinbase transaction as the first transaction
                    block_transactions = [coinbase_tx] + transactions
                    logging.debug(f"Added coinbase transaction to block")
                
                # Create new block
                block = Block(
                    version=1,
                    previous_hash=previous_block.hash,
                    transactions=block_transactions,
                    difficulty_target=self.blockchain.difficulty_target,
                    nonce=0,
                    timestamp=int(time.time()),
                    index=next_index  # Use the next sequential index
                )
            
            # Mine the block
            logging.info(f"Mining block {block.index} with {len(block.transactions)} transactions")
            while True:
                block.nonce += 1
                block.update_hash()
                if int(block.hash, 16) < 2**256 - block.difficulty_target:
                    break
            
            # Add block to chain
            if self.blockchain.add_block(block):
                logging.info(f"Mined block {block.index} with hash {block.hash}")
                return block
            else:
                logging.error(f"Failed to add mined block {block.index}")
                return None
                
        except Exception as e:
            logging.error(f"Error mining block: {e}")
            return None
