import sqlite3
import json
import logging
import os
import hashlib
import time
from Block import Block

class SQLiteStorage:
    def __init__(self, db_path="blockchain.db"):
        """Initialize SQLite storage"""
        # Set default database path
        if db_path == "blockchain.db":
            home_dir = os.path.expanduser("~")
            db_path = os.path.join(home_dir, ".local", "share", "sterlingx", "blockchain.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
        self.db_path = db_path
        self.cache = {}  # Block hash -> Block object
        self.cache_size = 10000  # Maximum number of blocks to keep in cache (increased from 1000)
        
        # Create database and tables if they don't exist
        self._init_db()
        
    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create blocks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS blocks (
                    hash TEXT PRIMARY KEY,
                    version INTEGER,
                    previous_hash TEXT,
                    transactions TEXT,
                    difficulty_target INTEGER,
                    nonce INTEGER,
                    timestamp INTEGER,
                    block_index INTEGER,
                    merkle_root TEXT,
                    checksum TEXT
                )
            ''')
            
            # Create indexes for blocks table
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_blocks_previous_hash ON blocks(previous_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_blocks_timestamp ON blocks(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_blocks_block_index ON blocks(block_index)')
            
            # Create block_backups table for redundancy
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS block_backups (
                    hash TEXT PRIMARY KEY,
                    block_data TEXT,
                    checksum TEXT,
                    backup_timestamp INTEGER
                )
            ''')
            
            # Create indexes for block_backups table
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_block_backups_timestamp ON block_backups(backup_timestamp)')
            
            # Create chain_metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chain_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    checksum TEXT
                )
            ''')
            
            # Create indexes for chain_metadata table
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chain_metadata_key ON chain_metadata(key)')
            
            # Create transactions table for better querying
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    txid TEXT PRIMARY KEY,
                    block_hash TEXT,
                    sender_address TEXT,
                    timestamp INTEGER,
                    fee REAL,
                    FOREIGN KEY (block_hash) REFERENCES blocks(hash)
                )
            ''')
            
            # Create indexes for transactions table
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_block_hash ON transactions(block_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_sender_address ON transactions(sender_address)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp)')
            
            # Create transaction_outputs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transaction_outputs (
                    txid TEXT,
                    output_index INTEGER,
                    address TEXT,
                    amount REAL,
                    spent BOOLEAN DEFAULT 0,
                    PRIMARY KEY (txid, output_index),
                    FOREIGN KEY (txid) REFERENCES transactions(txid)
                )
            ''')
            
            # Create indexes for transaction_outputs table
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transaction_outputs_address ON transaction_outputs(address)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transaction_outputs_spent ON transaction_outputs(spent)')
            
            # Create transaction_inputs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transaction_inputs (
                    txid TEXT,
                    input_index INTEGER,
                    prev_txid TEXT,
                    prev_output_index INTEGER,
                    PRIMARY KEY (txid, input_index),
                    FOREIGN KEY (txid) REFERENCES transactions(txid),
                    FOREIGN KEY (prev_txid, prev_output_index) REFERENCES transaction_outputs(txid, output_index)
                )
            ''')
            
            # Create indexes for transaction_inputs table
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transaction_inputs_prev_txid ON transaction_inputs(prev_txid)')
            
            conn.commit()
            
    def _calculate_checksum(self, data):
        """Calculate checksum for data"""
        if isinstance(data, str):
            return hashlib.sha256(data.encode()).hexdigest()
        return hashlib.sha256(json.dumps(data).encode()).hexdigest()
            
    def save_block(self, block):
        """Save a block to SQLite database with backup"""
        try:
            # Convert block to dictionary
            block_dict = block.to_dict() if hasattr(block, 'to_dict') else block
            
            # Calculate checksums
            block_data = json.dumps(block_dict)
            block_checksum = self._calculate_checksum(block_data)
            
            # Store in database with transaction for atomicity
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Begin transaction
                cursor.execute('BEGIN TRANSACTION')
                
                try:
                    # Save to main blocks table
                    cursor.execute('''
                        INSERT OR REPLACE INTO blocks 
                        (hash, version, previous_hash, transactions, difficulty_target, nonce, timestamp, block_index, merkle_root, checksum)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        block.hash,
                        block.version,
                        block.previous_hash,
                        json.dumps(block.transactions),
                        block.difficulty_target,
                        block.nonce,
                        block.timestamp,
                        block.index,
                        block.merkle_root,
                        block_checksum
                    ))
                    
                    # Save transactions
                    for tx in block.transactions:
                        # Save transaction
                        cursor.execute('''
                            INSERT OR REPLACE INTO transactions 
                            (txid, block_hash, sender_address, timestamp, fee)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (
                            tx['txid'],
                            block.hash,
                            tx.get('sender_address'),
                            tx.get('timestamp'),
                            tx.get('fee', 0)
                        ))
                        
                        # Save transaction outputs
                        for i, output in enumerate(tx.get('outputs', [])):
                            cursor.execute('''
                                INSERT OR REPLACE INTO transaction_outputs 
                                (txid, output_index, address, amount)
                                VALUES (?, ?, ?, ?)
                            ''', (
                                tx['txid'],
                                i,
                                output.get('address'),
                                output.get('amount')
                            ))
                            
                        # Save transaction inputs
                        for i, input_tx in enumerate(tx.get('inputs', [])):
                            cursor.execute('''
                                INSERT OR REPLACE INTO transaction_inputs 
                                (txid, input_index, prev_txid, prev_output_index)
                                VALUES (?, ?, ?, ?)
                            ''', (
                                tx['txid'],
                                i,
                                input_tx.get('txid'),
                                input_tx.get('output_index')
                            ))
                    
                    # Save backup
                    cursor.execute('''
                        INSERT OR REPLACE INTO block_backups 
                        (hash, block_data, checksum, backup_timestamp)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        block.hash,
                        block_data,
                        block_checksum,
                        int(time.time())
                    ))
                    
                    # Commit transaction
                    conn.commit()
                    
                except Exception as e:
                    # Rollback on error
                    conn.rollback()
                    raise e
                
            # Add to cache
            self.cache[block.hash] = block
            if len(self.cache) > self.cache_size:
                # Remove oldest block from cache
                self.cache.pop(next(iter(self.cache)))
                
        except Exception as e:
            logging.error(f"Error saving block {block.hash[:8]}: {e}")
            
    def load_block(self, block_hash):
        """Load a block from SQLite database with corruption recovery"""
        try:
            # Check cache first
            if block_hash in self.cache:
                return self.cache[block_hash]
                
            # Load from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Try to load from main table first
                cursor.execute('SELECT * FROM blocks WHERE hash = ?', (block_hash,))
                row = cursor.fetchone()
                
                if row:
                    # Verify checksum
                    stored_checksum = row[9]  # checksum column
                    block_data = {
                        'version': row[1],
                        'previous_hash': row[2],
                        'transactions': json.loads(row[3]),
                        'difficulty_target': row[4],
                        'nonce': row[5],
                        'timestamp': row[6],
                        'index': row[7],
                        'merkle_root': row[8],
                        'hash': row[0]
                    }
                    calculated_checksum = self._calculate_checksum(block_data)
                    
                    if stored_checksum == calculated_checksum:
                        # Data is valid, create block object
                        block = Block(
                            version=row[1],
                            previous_hash=row[2],
                            transactions=json.loads(row[3]),
                            difficulty_target=row[4],
                            nonce=row[5],
                            timestamp=row[6],
                            index=row[7]
                        )
                        block.merkle_root = row[8]
                        block.hash = row[0]
                        
                        # Add to cache
                        self.cache[block_hash] = block
                        if len(self.cache) > self.cache_size:
                            self.cache.pop(next(iter(self.cache)))
                            
                        return block
                
                # If we get here, either the block wasn't found or was corrupted
                # Try to recover from backup
                cursor.execute('SELECT block_data, checksum FROM block_backups WHERE hash = ?', (block_hash,))
                backup_row = cursor.fetchone()
                
                if backup_row:
                    backup_data = backup_row[0]
                    backup_checksum = backup_row[1]
                    
                    # Verify backup checksum
                    if self._calculate_checksum(backup_data) == backup_checksum:
                        # Backup is valid, restore to main table
                        block_dict = json.loads(backup_data)
                        block = Block.from_dict(block_dict)
                        
                        # Save restored block
                        self.save_block(block)
                        
                        # Add to cache
                        self.cache[block_hash] = block
                        if len(self.cache) > self.cache_size:
                            self.cache.pop(next(iter(self.cache)))
                            
                        logging.info(f"Recovered block {block_hash[:8]} from backup")
                        return block
                
                return None
                
        except Exception as e:
            logging.error(f"Error loading block {block_hash[:8]}: {e}")
            return None
            
    def delete_block(self, block_hash):
        """Delete a block from storage"""
        try:
            # Remove from cache
            self.cache.pop(block_hash, None)
            
            # Delete from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM blocks WHERE hash = ?', (block_hash,))
                cursor.execute('DELETE FROM block_backups WHERE hash = ?', (block_hash,))
                conn.commit()
                
        except Exception as e:
            logging.error(f"Error deleting block {block_hash[:8]}: {e}")
            
    def save_chain_metadata(self, key, value):
        """Save chain metadata to SQLite database"""
        try:
            # Convert value to JSON string if it's not already a string
            if not isinstance(value, str):
                value = json.dumps(value)
            
            # Calculate checksum
            checksum = self._calculate_checksum(value)
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO chain_metadata 
                    (key, value, checksum)
                    VALUES (?, ?, ?)
                ''', (key, value, checksum))
                conn.commit()
                
        except Exception as e:
            logging.error(f"Error saving chain metadata for key {key}: {e}")
            
    def load_chain_metadata(self, key):
        """Load chain metadata from SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT value, checksum FROM chain_metadata WHERE key = ?', (key,))
                row = cursor.fetchone()
                
                if row:
                    value, stored_checksum = row
                    
                    # Verify checksum
                    calculated_checksum = self._calculate_checksum(value)
                    if stored_checksum == calculated_checksum:
                        # Try to parse JSON if possible
                        try:
                            return json.loads(value)
                        except json.JSONDecodeError:
                            return value
                            
                return None
                
        except Exception as e:
            logging.error(f"Error loading chain metadata for key {key}: {e}")
            return None
            
    def save_blocks_batch(self, blocks):
        """Save multiple blocks in a single transaction"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('BEGIN TRANSACTION')
                
                try:
                    for block in blocks:
                        # Convert block to dictionary
                        block_dict = block.to_dict() if hasattr(block, 'to_dict') else block
                        
                        # Calculate checksums
                        block_data = json.dumps(block_dict)
                        block_checksum = self._calculate_checksum(block_data)
                        
                        # Save to main blocks table
                        cursor.execute('''
                            INSERT OR REPLACE INTO blocks 
                            (hash, version, previous_hash, transactions, difficulty_target, nonce, timestamp, block_index, merkle_root, checksum)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            block.hash,
                            block.version,
                            block.previous_hash,
                            json.dumps(block.transactions),
                            block.difficulty_target,
                            block.nonce,
                            block.timestamp,
                            block.index,
                            block.merkle_root,
                            block_checksum
                        ))
                        
                        # Save transactions in batch
                        tx_data = []
                        output_data = []
                        input_data = []
                        
                        for tx in block.transactions:
                            # Prepare transaction data
                            tx_data.append((
                                tx['txid'],
                                block.hash,
                                tx.get('sender_address'),
                                tx.get('timestamp'),
                                tx.get('fee', 0)
                            ))
                            
                            # Prepare output data
                            for i, output in enumerate(tx.get('outputs', [])):
                                output_data.append((
                                    tx['txid'],
                                    i,
                                    output.get('address'),
                                    output.get('amount')
                                ))
                                
                            # Prepare input data
                            for i, input_tx in enumerate(tx.get('inputs', [])):
                                input_data.append((
                                    tx['txid'],
                                    i,
                                    input_tx.get('txid'),
                                    input_tx.get('output_index')
                                ))
                        
                        # Execute batch inserts
                        if tx_data:
                            cursor.executemany('''
                                INSERT OR REPLACE INTO transactions 
                                (txid, block_hash, sender_address, timestamp, fee)
                                VALUES (?, ?, ?, ?, ?)
                            ''', tx_data)
                            
                        if output_data:
                            cursor.executemany('''
                                INSERT OR REPLACE INTO transaction_outputs 
                                (txid, output_index, address, amount)
                                VALUES (?, ?, ?, ?)
                            ''', output_data)
                            
                        if input_data:
                            cursor.executemany('''
                                INSERT OR REPLACE INTO transaction_inputs 
                                (txid, input_index, prev_txid, prev_output_index)
                                VALUES (?, ?, ?, ?)
                            ''', input_data)
                        
                        # Save backup
                        cursor.execute('''
                            INSERT OR REPLACE INTO block_backups 
                            (hash, block_data, checksum, backup_timestamp)
                            VALUES (?, ?, ?, ?)
                        ''', (
                            block.hash,
                            block_data,
                            block_checksum,
                            int(time.time())
                        ))
                        
                        # Add to cache
                        self.cache[block.hash] = block
                        if len(self.cache) > self.cache_size:
                            self.cache.pop(next(iter(self.cache)))
                    
                    conn.commit()
                    
                except Exception as e:
                    conn.rollback()
                    raise e
                    
        except Exception as e:
            logging.error(f"Error saving blocks batch: {e}")
            
    def load_blocks_batch(self, block_hashes):
        """Load multiple blocks in a single query"""
        try:
            blocks = []
            missing_hashes = []
            
            # Check cache first
            for block_hash in block_hashes:
                if block_hash in self.cache:
                    blocks.append(self.cache[block_hash])
                else:
                    missing_hashes.append(block_hash)
                    
            if not missing_hashes:
                return blocks
                
            # Load remaining blocks from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Use parameterized query with multiple values
                placeholders = ','.join(['?' for _ in missing_hashes])
                cursor.execute(f'''
                    SELECT * FROM blocks 
                    WHERE hash IN ({placeholders})
                ''', missing_hashes)
                
                rows = cursor.fetchall()
                
                for row in rows:
                    # Verify checksum
                    stored_checksum = row[9]
                    block_data = {
                        'version': row[1],
                        'previous_hash': row[2],
                        'transactions': json.loads(row[3]),
                        'difficulty_target': row[4],
                        'nonce': row[5],
                        'timestamp': row[6],
                        'index': row[7],
                        'merkle_root': row[8],
                        'hash': row[0]
                    }
                    calculated_checksum = self._calculate_checksum(block_data)
                    
                    if stored_checksum == calculated_checksum:
                        block = Block(
                            version=row[1],
                            previous_hash=row[2],
                            transactions=json.loads(row[3]),
                            difficulty_target=row[4],
                            nonce=row[5],
                            timestamp=row[6],
                            index=row[7]
                        )
                        block.merkle_root = row[8]
                        block.hash = row[0]
                        
                        blocks.append(block)
                        self.cache[block.hash] = block
                        if len(self.cache) > self.cache_size:
                            self.cache.pop(next(iter(self.cache)))
                            
            return blocks
            
        except Exception as e:
            logging.error(f"Error loading blocks batch: {e}")
            return []
            
    def delete_blocks_batch(self, block_hashes):
        """Delete multiple blocks in a single transaction"""
        try:
            # Remove from cache
            for block_hash in block_hashes:
                self.cache.pop(block_hash, None)
                
            # Delete from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('BEGIN TRANSACTION')
                
                try:
                    # Use parameterized query with multiple values
                    placeholders = ','.join(['?' for _ in block_hashes])
                    cursor.execute(f'''
                        DELETE FROM blocks 
                        WHERE hash IN ({placeholders})
                    ''', block_hashes)
                    
                    cursor.execute(f'''
                        DELETE FROM block_backups 
                        WHERE hash IN ({placeholders})
                    ''', block_hashes)
                    
                    conn.commit()
                    
                except Exception as e:
                    conn.rollback()
                    raise e
                    
        except Exception as e:
            logging.error(f"Error deleting blocks batch: {e}")
            
    def get_utxos_batch(self, addresses):
        """Get UTXOs for multiple addresses in a single query"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Use parameterized query with multiple values
                placeholders = ','.join(['?' for _ in addresses])
                cursor.execute(f'''
                    SELECT t.txid, t.block_hash, t.sender_address, t.timestamp, t.fee,
                           o.output_index, o.address, o.amount, o.spent
                    FROM transactions t
                    JOIN transaction_outputs o ON t.txid = o.txid
                    WHERE o.address IN ({placeholders}) AND o.spent = 0
                ''', addresses)
                
                utxos = []
                for row in cursor.fetchall():
                    utxo = {
                        'txid': row[0],
                        'block_hash': row[1],
                        'sender_address': row[2],
                        'timestamp': row[3],
                        'fee': row[4],
                        'output_index': row[5],
                        'address': row[6],
                        'amount': row[7],
                        'spent': row[8]
                    }
                    utxos.append(utxo)
                    
                return utxos
                
        except Exception as e:
            logging.error(f"Error getting UTXOs batch: {e}")
            return []
            
    def mark_utxos_spent_batch(self, utxos):
        """Mark multiple UTXOs as spent in a single transaction"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('BEGIN TRANSACTION')
                
                try:
                    # Prepare data for batch update
                    spent_data = [(utxo['txid'], utxo['output_index']) for utxo in utxos]
                    
                    # Execute batch update
                    cursor.executemany('''
                        UPDATE transaction_outputs 
                        SET spent = 1 
                        WHERE txid = ? AND output_index = ?
                    ''', spent_data)
                    
                    conn.commit()
                    
                except Exception as e:
                    conn.rollback()
                    raise e
                    
        except Exception as e:
            logging.error(f"Error marking UTXOs spent batch: {e}") 