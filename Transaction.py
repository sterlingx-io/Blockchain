import hashlib
import time
import ecdsa
import logging
import json
import random

class Transaction:
    # Helper functions for 8 decimal precision
    @staticmethod
    def format_amount(amount):
        """Format amount to 8 decimal places"""
        return float(f"{float(amount):.8f}")

    @staticmethod
    def validate_amount(amount):
        """Validate that amount has 8 decimal places"""
        try:
            formatted = Transaction.format_amount(amount)
            return abs(float(amount) - formatted) < 1e-9
        except (ValueError, TypeError):
            return False

    @staticmethod
    def round_amount(amount):
        """Round amount to 8 decimal places"""
        return round(float(amount), 8)

    # Constants for fee calculation
    MIN_FEE = 0.00000001  # Minimum fee in base units (8 decimals)
    FEE_PER_BYTE = 0.00000001  # Fee per byte in base units (8 decimals)
    MIN_TRANSACTION_SIZE = 100  # Minimum transaction size in bytes
    MAX_TRANSACTION_SIZE = 100000  # Maximum transaction size in bytes
    TARGET_BLOCKSIZE = 1000000  # Target block size in bytes
    FEE_ADJUSTMENT_INTERVAL = 144  # Adjust fees every 144 blocks
    MAX_FEE_MULTIPLIER = 10.0  # Maximum fee multiplier
    MIN_FEE_MULTIPLIER = 0.1  # Minimum fee multiplier
    DUST_LIMIT = 0.00000001  # Minimum output amount (8 decimals)
    MIN_FEE_RATE = 0.00000001  # Minimum fee rate per byte (8 decimals)
    MAX_TX_SIZE = 100000  # Maximum transaction size in bytes
    PRIORITY_THRESHOLDS = {
        "high": 0.00000100,  # High priority fee rate (8 decimals)
        "medium": 0.00000050,  # Medium priority fee rate (8 decimals)
        "low": 0.00000001  # Low priority fee rate (8 decimals)
    }
    RBF_THRESHOLD = 1.1  # Minimum fee multiplier for RBF
    MAX_RBF_ATTEMPTS = 3  # Maximum number of RBF attempts
    TX_EXPIRATION = 3600  # Transaction expiration time in seconds
    NETWORK_ID = "mainnet"  # Network identifier
    CHAIN_ID = "1"  # Chain identifier

    def __init__(self, inputs, outputs, fee=None, nonce=None, timestamp=None, network_id=None, chain_id=None, rbf=False, rbf_attempt=0, sender_address=None, public_key=None, signature=None):
        self.inputs = inputs
        self.outputs = outputs
        self.timestamp = timestamp or int(time.time())
        self.nonce = nonce or random.randint(0, 2**32-1)
        self.network_id = network_id or self.NETWORK_ID
        self.chain_id = chain_id or self.CHAIN_ID
        self.rbf = rbf
        self.rbf_attempt = rbf_attempt
        self.expiration_time = self.timestamp + self.TX_EXPIRATION
        
        # Set sender_address, respecting the provided value for genesis transactions
        self.sender_address = sender_address if sender_address else (inputs[0]["address"] if inputs else "COINBASE")
        self.receiver = outputs[0]["address"] if outputs else None
        self.amount = outputs[0]["amount"] if outputs else 0
        
        # For genesis transactions, set fee to 0
        if self.sender_address == "GENESIS":
            self.fee = 0
        else:
            # Set fee before calculating size
            self.fee = self.format_amount(fee) if fee is not None else self.MIN_FEE
        
        # Initialize signature and public key
        self.signature = signature
        self.public_key = public_key
        
        # Calculate size
        self.size = self.calculate_size()
        
        # Update fee if dynamic and not genesis
        if fee is None and self.sender_address != "GENESIS":
            self.fee = self.calculate_dynamic_fee()
            
        # Validate RBF parameters
        if self.rbf and self.rbf_attempt >= self.MAX_RBF_ATTEMPTS:
            raise ValueError("Maximum RBF attempts exceeded")
            
        # Calculate transaction ID
        self.txid = self.calculate_txid()
        
        # Set priority
        self.priority = self.get_priority()

    def calculate_size(self):
        """Calculate transaction size in bytes"""
        # Approximate size calculation
        size = 0
        size += len(str(self.sender_address))
        size += len(str(self.receiver))
        size += len(str(self.amount))
        size += len(str(self.fee))
        size += len(str(self.timestamp))
        size += len(str(self.nonce))
        size += len(str(self.network_id))
        size += len(str(self.chain_id))
        
        for input_tx in self.inputs:
            size += len(str(input_tx['txid']))
            size += len(str(input_tx['address']))
            size += len(str(input_tx['amount']))
        
        for output in self.outputs:
            size += len(str(output['address']))
            size += len(str(output['amount']))
        
        if self.signature:
            size += len(self.signature)
        
        return size

    def calculate_dynamic_fee(self):
        """Calculate dynamic fee based on network congestion"""
        try:
            # Get recent block sizes
            recent_blocks = self.get_recent_blocks()
            if not recent_blocks:
                return self.MIN_FEE
            
            # Calculate average block size
            avg_block_size = sum(block['size'] for block in recent_blocks) / len(recent_blocks)
            
            # Calculate congestion ratio
            congestion_ratio = avg_block_size / self.TARGET_BLOCKSIZE
            
            # Calculate dynamic fee multiplier
            fee_multiplier = min(
                max(congestion_ratio, self.MIN_FEE_MULTIPLIER),
                self.MAX_FEE_MULTIPLIER
            )
            
            # Calculate base fee
            base_fee = self.size * self.FEE_PER_BYTE
            
            # Apply dynamic multiplier
            dynamic_fee = base_fee * fee_multiplier
            
            # Ensure minimum fee
            return max(dynamic_fee, self.MIN_FEE)
        except Exception as e:
            logging.error(f"Error calculating dynamic fee: {e}")
            return self.MIN_FEE

    def get_priority(self):
        """Get transaction priority based on fee rate"""
        fee_rate = self.fee / self.size
        if fee_rate >= self.PRIORITY_THRESHOLDS["high"]:
            return "high"
        elif fee_rate >= self.PRIORITY_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "low"

    def can_replace(self, old_tx):
        """Check if this transaction can replace an old one"""
        if not self.rbf or not old_tx.rbf:
            return False
            
        if self.rbf_attempt >= self.MAX_RBF_ATTEMPTS:
            return False
            
        # Check if new fee is high enough
        if self.fee < old_tx.fee * self.RBF_THRESHOLD:
            return False
            
        # Check if inputs are the same
        if len(self.inputs) != len(old_tx.inputs):
            return False
            
        for new_input, old_input in zip(self.inputs, old_tx.inputs):
            if new_input['txid'] != old_input['txid'] or new_input['index'] != old_input['index']:
                return False
                
        return True

    def bump_fee(self, multiplier=1.1):
        """Bump transaction fee for RBF"""
        if not self.rbf or self.rbf_attempt >= self.MAX_RBF_ATTEMPTS:
            raise ValueError("Cannot bump fee for non-RBF transaction or max attempts reached")
            
        new_fee = self.fee * multiplier
        return Transaction(
            inputs=self.inputs,
            outputs=self.outputs,
            fee=new_fee,
            nonce=self.nonce,
            timestamp=self.timestamp,
            network_id=self.network_id,
            chain_id=self.chain_id,
            rbf=True,
            rbf_attempt=self.rbf_attempt + 1
        )

    def calculate_txid(self):
        """Calculate transaction ID including all fields except signature"""
        if self.sender_address == "GENESIS":
            # For genesis transactions, include all necessary fields
            tx_dict = {
                "sender_address": self.sender_address,
                "receiver": self.receiver,
                "amount": self.amount,
                "fee": self.fee,
                "timestamp": self.timestamp,
                "nonce": self.nonce,
                "network_id": self.network_id,
                "chain_id": self.chain_id,
                "outputs": self.outputs
            }
            tx_string = json.dumps(tx_dict, sort_keys=True)
            return hashlib.sha256(tx_string.encode()).hexdigest()
        elif self.sender_address == "COINBASE":
            tx_string = f"{self.network_id}{self.chain_id}{self.sender_address}{self.receiver}{self.amount}{self.fee}{self.timestamp}{self.nonce}"
            return hashlib.sha256(tx_string.encode()).hexdigest()
        else:
            # Convert inputs to string
            inputs_str = json.dumps(self.inputs, sort_keys=True)
            tx_string = f"{self.network_id}{self.chain_id}{self.sender_address}{self.receiver}{self.amount}{self.fee}{self.timestamp}{self.nonce}{inputs_str}"
            return hashlib.sha256(tx_string.encode()).hexdigest()

    def sign_transaction(self, private_key):
        """Sign a transaction with a private key"""
        try:
            # Create transaction dictionary for signing
            tx_dict = {
                'txid': self.txid,
                'sender_address': self.sender_address,
                'receiver': self.receiver,
                'amount': self.amount,
                'fee': self.fee,
                'inputs': self.inputs,
                'outputs': self.outputs,
                'timestamp': self.timestamp,
                'nonce': self.nonce,
                'network_id': self.network_id,
                'chain_id': self.chain_id,
                'rbf': self.rbf,
                'rbf_attempt': self.rbf_attempt
            }
            
            # Convert to bytes using UTF-8 encoding
            tx_bytes = json.dumps(tx_dict, sort_keys=True).encode('utf-8')
            
            # Sign transaction using raw format
            signature = private_key.sign(tx_bytes, hashfunc=hashlib.sha256, sigencode=ecdsa.util.sigencode_string)
            
            # Verify signature length
            if len(signature) != 64:
                logging.error(f"Invalid signature length: {len(signature)} bytes")
                return None
            
            # Store signature in hex format
            self.signature = signature.hex()
            
            # Store public key
            self.public_key = private_key.get_verifying_key().to_string().hex()
            
            return self.signature
        except Exception as e:
            logging.error(f"Error signing transaction: {e}")
            return None

    def verify_signature(self):
        """Verify the transaction signature"""
        try:
            # Skip signature verification for special transaction types
            if self.sender_address in ["COINBASE", "FEES", "GENESIS"]:
                return True
                
            # Create transaction dictionary in the exact same format as signing
            tx_dict = {
                'txid': self.txid,
                'sender_address': self.sender_address,
                'receiver': self.receiver,
                'amount': self.amount,
                'fee': self.fee,
                'inputs': self.inputs,
                'outputs': self.outputs,
                'timestamp': self.timestamp,
                'nonce': self.nonce,
                'network_id': self.network_id,
                'chain_id': self.chain_id,
                'rbf': self.rbf,
                'rbf_attempt': self.rbf_attempt
            }
            
            # Convert to bytes using UTF-8 encoding
            tx_bytes = json.dumps(tx_dict, sort_keys=True).encode('utf-8')
            
            # Verify signature using raw signature format
            public_key = ecdsa.VerifyingKey.from_string(bytes.fromhex(self.public_key), curve=ecdsa.SECP256k1)
            signature_bytes = bytes.fromhex(self.signature)
            return public_key.verify(signature_bytes, tx_bytes, hashfunc=hashlib.sha256, sigdecode=ecdsa.util.sigdecode_string)
            
        except Exception as e:
            print(f"Signature verification failed: {str(e)}")
            return False

    def to_dict(self):
        """Convert transaction to dictionary"""
        return {
            'txid': self.txid,
            'sender_address': self.sender_address,
            'receiver': self.receiver,
            'amount': self.amount,
            'fee': self.fee,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'signature': self.signature,
            'public_key': self.public_key,
            'timestamp': self.timestamp,
            'nonce': self.nonce,
            'network_id': self.network_id,
            'chain_id': self.chain_id,
            'expiration_time': self.expiration_time,
            'priority': self.priority,
            'size': self.size,
            'rbf': self.rbf,
            'rbf_attempt': self.rbf_attempt
        }
