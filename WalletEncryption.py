import os
import json
import base64
import logging
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

class WalletEncryption:
    def __init__(self, passphrase=None):
        """Initialize wallet encryption with optional passphrase"""
        self.passphrase = passphrase.encode() if passphrase else None
        self.salt = None
        self.key = None
        if passphrase:
            self._derive_key()

    def _derive_key(self, salt=None):
        """Derive encryption key from passphrase using PBKDF2"""
        if not self.passphrase:
            raise ValueError("Passphrase required for key derivation")
            
        # Use provided salt or generate new one
        self.salt = salt if salt else os.urandom(16)
            
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
            backend=default_backend()
        )
        self.key = kdf.derive(self.passphrase)

    def encrypt_wallet(self, wallet_data):
        """Encrypt wallet data using AES-256-GCM"""
        if not self.key:
            raise ValueError("Encryption key not available")
            
        # Convert wallet data to JSON string if it's not already a string
        if not isinstance(wallet_data, str):
            wallet_data = json.dumps(wallet_data)
        
        # Generate random nonce
        nonce = os.urandom(12)
        
        # Create AES-GCM cipher
        aesgcm = AESGCM(self.key)
        
        # Encrypt data
        ciphertext = aesgcm.encrypt(nonce, wallet_data.encode(), None)
        
        # Return encrypted data with salt and nonce
        return {
            'salt': base64.b64encode(self.salt).decode(),
            'nonce': base64.b64encode(nonce).decode(),
            'ciphertext': base64.b64encode(ciphertext).decode()
        }

    def decrypt_wallet(self, encrypted_data):
        """Decrypt wallet data using AES-256-GCM"""
        try:
            if not isinstance(encrypted_data, dict):
                raise ValueError("Invalid encrypted data format")
                
            if not all(k in encrypted_data for k in ['salt', 'nonce', 'ciphertext']):
                raise ValueError("Missing required encryption fields")
                
            # Decode base64 data
            salt = base64.b64decode(encrypted_data['salt'])
            nonce = base64.b64decode(encrypted_data['nonce'])
            ciphertext = base64.b64decode(encrypted_data['ciphertext'])
            
            # Derive key with the same salt
            self._derive_key(salt)
            
            # Create AES-GCM cipher
            aesgcm = AESGCM(self.key)
            
            # Decrypt data
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            
            # Return decrypted data
            return plaintext.decode()
            
        except Exception as e:
            logging.error(f"Error decrypting wallet: {e}")
            raise

    def save_encrypted_wallet(self, wallet_data, filename):
        """Save encrypted wallet to file"""
        try:
            encrypted_data = self.encrypt_wallet(wallet_data)
            with open(filename, 'w') as f:
                json.dump(encrypted_data, f, indent=2)
            logging.info(f"💾 Encrypted wallet saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving encrypted wallet: {e}")
            raise

    def load_encrypted_wallet(self, filename):
        """Load and decrypt wallet from file"""
        try:
            with open(filename, 'r') as f:
                encrypted_data = json.load(f)
            wallet_data = self.decrypt_wallet(encrypted_data)
            logging.info(f"📂 Encrypted wallet loaded from {filename}")
            return wallet_data
        except Exception as e:
            logging.error(f"Error loading encrypted wallet: {e}")
            raise 