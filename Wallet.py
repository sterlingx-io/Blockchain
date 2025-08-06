import hashlib
import json
import os
import sys
import ecdsa
from mnemonic import Mnemonic
import logging
from WalletEncryption import WalletEncryption

class Wallet:
    def __init__(self, seed_phrase=None):
        self.mnemo = Mnemonic("english")
        if seed_phrase:
            self.seed_phrase = seed_phrase
            seed = self.mnemo.to_seed(seed_phrase)
            self.private_key = ecdsa.SigningKey.from_string(bytes.fromhex(seed.hex()[:64]), curve=ecdsa.SECP256k1)
        else:
            self.seed_phrase = self.mnemo.generate(strength=256)
            seed = self.mnemo.to_seed(self.seed_phrase)
            self.private_key = ecdsa.SigningKey.from_string(bytes.fromhex(seed.hex()[:64]), curve=ecdsa.SECP256k1)
        self.public_key = self.private_key.verifying_key
        self.address = self.generate_address()

    def generate_address(self):
        pubkey_hex = self.public_key.to_string().hex()
        return hashlib.sha256(pubkey_hex.encode()).hexdigest()[:32]

    def get_balance(self, blockchain):
        return blockchain.get_balance(self.address)

    def to_dict(self):
        return {
            "private_key": self.private_key.to_string().hex(),
            "public_key": self.public_key.to_string().hex(),
            "address": self.address,
            "name": getattr(self, 'name', None)
        }

    @staticmethod
    def from_private_key(private_key_hex):
        wallet = Wallet()
        wallet.private_key = ecdsa.SigningKey.from_string(bytes.fromhex(private_key_hex), curve=ecdsa.SECP256k1)
        wallet.public_key = wallet.private_key.verifying_key
        wallet.address = wallet.generate_address()
        return wallet

    @staticmethod
    def from_seed_phrase(seed_phrase):
        return Wallet(seed_phrase=seed_phrase)

    def sign_transaction(self, transaction):
        """Sign a transaction"""
        try:
            # Convert transaction to bytes using consistent fields
            tx_dict = {
                'txid': transaction['txid'],
                'sender_address': transaction['sender_address'],
                'receiver': transaction['receiver'],
                'amount': transaction['amount'],
                'fee': transaction['fee'],
                'inputs': transaction['inputs'],
                'outputs': transaction['outputs'],
                'timestamp': transaction['timestamp'],
                'nonce': transaction['nonce'],
                'network_id': transaction['network_id'],
                'chain_id': transaction['chain_id'],
                'rbf': transaction['rbf'],
                'rbf_attempt': transaction['rbf_attempt']
            }
            tx_data = json.dumps(tx_dict, sort_keys=True).encode('utf-8')
            
            # Sign transaction using raw format
            signature = self.private_key.sign(tx_data, hashfunc=hashlib.sha256, sigencode=ecdsa.util.sigencode_string)
            
            # Verify signature length
            if len(signature) != 64:
                logging.error(f"Invalid signature length: {len(signature)} bytes")
                return None
            
            # Convert to hex and verify it's 128 characters (64 bytes)
            signature_hex = signature.hex()
            if len(signature_hex) != 128:
                logging.error(f"Invalid signature hex length: {len(signature_hex)} characters")
                return None
            
            return signature_hex
        except Exception as e:
            logging.error(f"Error signing transaction: {e}")
            return None

    def verify_signature(self, signature_hex, transaction):
        """Verify a transaction signature"""
        try:
            # Convert transaction to bytes using consistent fields
            tx_dict = {
                'txid': transaction['txid'],
                'sender_address': transaction['sender_address'],
                'receiver': transaction['receiver'],
                'amount': transaction['amount'],
                'fee': transaction['fee'],
                'inputs': transaction['inputs'],
                'outputs': transaction['outputs'],
                'timestamp': transaction['timestamp'],
                'nonce': transaction['nonce'],
                'network_id': transaction['network_id'],
                'chain_id': transaction['chain_id'],
                'rbf': transaction['rbf'],
                'rbf_attempt': transaction['rbf_attempt']
            }
            tx_data = json.dumps(tx_dict, sort_keys=True).encode('utf-8')
            
            # Convert hex signature back to bytes
            signature = bytes.fromhex(signature_hex)
            
            # Verify signature using raw format
            self.public_key.verify(signature, tx_data, hashfunc=hashlib.sha256, sigdecode=ecdsa.util.sigdecode_string)
            return True
        except Exception as e:
            logging.error(f"Error verifying signature: {e}")
            return False

class WalletManager:
    def __init__(self, filename="wallets.json", passphrase=None):
        # Set default wallet path
        if filename == "wallets.json":
            home_dir = os.path.expanduser("~")
            filename = os.path.join(home_dir, ".local", "share", "sterlingx", "wallets.json")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
        self.filename = filename
        self.wallets = {}
        self.encryption = WalletEncryption(passphrase)
        self.load_wallets()

    def generate_unique_name(self, base_name):
        """Generate a unique wallet name by appending a counter if needed."""
        name = f"wallet_{base_name}"
        counter = 0
        original_name = name
        self.load_wallets()
        logging.debug(f"Checking uniqueness for base name: {original_name}, current wallets: {list(self.wallets.keys())}")
        while name in self.wallets:
            counter += 1
            name = f"{original_name}{counter}"
            if len(name) > 32:
                name = name[:31] + str(counter)[-1]
            logging.debug(f"Generated unique name attempt: {name}")
        return name

    def create_wallet(self, name):
        logging.debug(f"Validating wallet name: {name}, length: {len(name)}, isalnum: {name.isalnum()}")
        if not name.isalnum() or len(name) > 32:
            logging.error(f"Invalid wallet name: {name}")
            return None
        if name in self.wallets:
            logging.warning(f"Wallet '{name}' already exists")
            return None
        wallet = Wallet()
        wallet.name = name
        self.wallets[name] = wallet.to_dict()
        self.save_wallets()
        logging.info(f"Created wallet: {name}, address: {wallet.address}")
        return wallet

    def get_wallet(self, name):
        wallet_data = self.wallets.get(name)
        if not wallet_data:
            return None
        return Wallet.from_private_key(wallet_data["private_key"])

    def import_wallet(self, name, seed_phrase):
        logging.debug(f"Importing wallet with name: {name}")
        if not name.isalnum() or len(name) > 32:
            logging.error(f"Invalid wallet name: {name}")
            return None
        if name in self.wallets:
            logging.warning(f"Wallet '{name}' already exists")
            return None
        mnemo = Mnemonic("english")
        if not mnemo.check(seed_phrase):
            logging.error(f"Invalid seed phrase for '{name}'")
            return None
        wallet = Wallet.from_seed_phrase(seed_phrase)
        wallet.name = name
        self.wallets[name] = wallet.to_dict()
        self.save_wallets()
        return wallet

    def save_wallets(self):
        try:
            if self.encryption.passphrase:
                self.encryption.save_encrypted_wallet(self.wallets, self.filename)
            else:
                with open(self.filename, 'w') as f:
                    json.dump(self.wallets, f, indent=2)
                logging.info(f"💾 Wallets saved to {self.filename}")
        except Exception as e:
            logging.error(f"Failed to save wallets: {e}")

    def load_wallets(self):
        try:
            if os.path.exists(self.filename):
                if self.encryption.passphrase:
                    self.wallets = self.encryption.load_encrypted_wallet(self.filename)
                else:
                    with open(self.filename, 'r') as f:
                        self.wallets = json.load(f)
                logging.info(f"📂 Wallets loaded from {self.filename}: {list(self.wallets.keys())}")
            else:
                self.wallets = {}
        except Exception as e:
            logging.error(f"Failed to load wallets: {e}")
            self.wallets = {}
