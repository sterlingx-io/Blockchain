import time
import logging
import heapq
import threading

class TransactionPool:
    def __init__(self, max_size=10000, max_mempool_size=100000000):
        self.transactions = {}  # txid -> transaction
        self.dependencies = {}  # txid -> set of dependent txids
        self.parents = {}       # txid -> set of parent txids
        self.max_size = max_size  # Maximum number of transactions
        self.max_mempool_size = max_mempool_size  # Maximum mempool size in bytes
        self.current_size = 0  # Current mempool size in bytes
        self.cleanup_interval = 300  # Cleanup every 5 minutes
        self.last_cleanup = time.time()
        self.priority_queues = {
            "high": [],
            "medium": [],
            "low": []
        }
        self.rbf_transactions = {}  # original_txid -> [replacement_txs]
        # Add locks for thread safety
        self.pool_lock = threading.RLock()  # Reentrant lock for transaction pool operations
        self.queue_locks = {
            "high": threading.Lock(),
            "medium": threading.Lock(),
            "low": threading.Lock()
        }
        self.rbf_lock = threading.Lock()  # Lock for RBF operations

    def add_transaction(self, transaction):
        """Add transaction to pool with proper prioritization"""
        try:
            # Check if transaction is valid
            if not transaction.validate():
                return False
                
            # Check if transaction is expired
            if time.time() > transaction.expiration_time:
                return False
            
            with self.pool_lock:
                # Check if transaction is replacing another one
                if transaction.rbf:
                    with self.rbf_lock:
                        for input_tx in transaction.inputs:
                            if input_tx['txid'] in self.transactions:
                                old_tx = self.transactions[input_tx['txid']]
                                if not transaction.can_replace(old_tx):
                                    return False
                                    
                                # Remove old transaction
                                self.remove_transaction(old_tx.txid)
                                
                                # Track RBF replacement
                                if old_tx.txid not in self.rbf_transactions:
                                    self.rbf_transactions[old_tx.txid] = []
                                self.rbf_transactions[old_tx.txid].append(transaction.txid)
                
                # Check if pool is full
                if len(self.transactions) >= self.max_size:
                    # Try to evict low priority transactions
                    if not self.evict_low_priority():
                        return False
                        
                # Check if mempool size limit is reached
                if self.current_size + transaction.size > self.max_mempool_size:
                    # Try to evict low priority transactions
                    if not self.evict_low_priority():
                        return False
                
                # Add transaction to pool
                self.transactions[transaction.txid] = transaction
                self.current_size += transaction.size
                
                # Add to priority queue
                priority = transaction.get_priority()
                with self.queue_locks[priority]:
                    heapq.heappush(self.priority_queues[priority], (-transaction.fee, transaction.txid, transaction))
                
                # Build dependency graph
                self._build_dependencies(transaction)
                
                return True
        except Exception as e:
            logging.error(f"Error adding transaction to pool: {e}")
            return False

    def _build_dependencies(self, tx):
        """Build dependency graph for transaction"""
        with self.pool_lock:
            # Skip coinbase transactions
            if tx.sender_address == "COINBASE":
                return
            
            # Initialize dependency sets
            self.dependencies[tx.txid] = set()
            self.parents[tx.txid] = set()
            
            # Check each input for dependencies
            for input_tx in tx.inputs:
                input_txid = input_tx['txid']
                
                # If input transaction is in pool, add dependency
                if input_txid in self.transactions:
                    self.dependencies[tx.txid].add(input_txid)
                    self.parents[input_txid].add(tx.txid)

    def evict_low_priority(self):
        """Evict low priority transactions to make space"""
        try:
            # Try to evict from low priority queue first
            with self.queue_locks["low"]:
                if self.priority_queues["low"]:
                    _, txid, _ = heapq.heappop(self.priority_queues["low"])
                    with self.pool_lock:
                        self.remove_transaction(txid)
                        return True
                    
            # Then try medium priority
            with self.queue_locks["medium"]:
                if self.priority_queues["medium"]:
                    _, txid, _ = heapq.heappop(self.priority_queues["medium"])
                    with self.pool_lock:
                        self.remove_transaction(txid)
                        return True
                    
            # Finally try high priority
            with self.queue_locks["high"]:
                if self.priority_queues["high"]:
                    _, txid, _ = heapq.heappop(self.priority_queues["high"])
                    with self.pool_lock:
                        self.remove_transaction(txid)
                        return True
                    
            return False
        except Exception as e:
            logging.error(f"Error evicting transactions: {e}")
            return False

    def remove_transaction(self, txid):
        """Remove transaction from pool"""
        try:
            with self.pool_lock:
                if txid in self.transactions:
                    transaction = self.transactions[txid]
                    self.current_size -= transaction.size
                    
                    # Remove from priority queue
                    priority = transaction.get_priority()
                    with self.queue_locks[priority]:
                        self.priority_queues[priority] = [
                            (fee, tid, tx) for fee, tid, tx in self.priority_queues[priority]
                            if tid != txid
                        ]
                        heapq.heapify(self.priority_queues[priority])
                    
                    del self.transactions[txid]
                    
                    # Clean up RBF tracking
                    with self.rbf_lock:
                        if txid in self.rbf_transactions:
                            del self.rbf_transactions[txid]
                            
                        for original_txid, replacements in list(self.rbf_transactions.items()):
                            if txid in replacements:
                                replacements.remove(txid)
                                if not replacements:
                                    del self.rbf_transactions[original_txid]
                                    
                    # Remove from dependency graph
                    if txid in self.dependencies:
                        for dep_txid in self.dependencies[txid]:
                            if dep_txid in self.parents:
                                self.parents[dep_txid].remove(txid)
                        del self.dependencies[txid]
                    
                    if txid in self.parents:
                        for parent_txid in self.parents[txid]:
                            if parent_txid in self.dependencies:
                                self.dependencies[parent_txid].remove(txid)
                        del self.parents[txid]
                    
                return True
        except Exception as e:
            logging.error(f"Error removing transaction from pool: {e}")
            return False

    def get_transactions_for_block(self, max_size=None):
        """Get transactions for new block with proper prioritization"""
        try:
            if max_size is None:
                max_size = self.max_mempool_size
                
            selected_txs = []
            current_size = 0
            
            # Process transactions in priority order
            for priority in ["high", "medium", "low"]:
                with self.queue_locks[priority]:
                    queue = self.priority_queues[priority].copy()
                    while queue and current_size < max_size:
                        _, txid, transaction = heapq.heappop(queue)
                        with self.pool_lock:
                            if txid in self.transactions:  # Check if transaction still exists
                                if current_size + transaction.size <= max_size:
                                    selected_txs.append(transaction)
                                    current_size += transaction.size
                                else:
                                    break
                                    
            return selected_txs
        except Exception as e:
            logging.error(f"Error getting transactions for block: {e}")
            return []

    def get_fee_estimate(self, priority="medium", blocks=1):
        """Get fee estimate for desired priority and confirmation time"""
        try:
            # Get recent transactions
            recent_txs = []
            for queue in self.priority_queues.values():
                recent_txs.extend([tx for _, _, tx in queue])
                
            if not recent_txs:
                return self.MIN_FEE_RATE
                
            # Sort by fee rate
            recent_txs.sort(key=lambda tx: tx.fee / tx.size, reverse=True)
            
            # Calculate target position based on desired blocks
            target_position = min(len(recent_txs) - 1, blocks * 10)
            
            if target_position < 0:
                return self.MIN_FEE_RATE
                
            # Get fee rate at target position
            target_tx = recent_txs[target_position]
            return target_tx.fee / target_tx.size
        except Exception as e:
            logging.error(f"Error getting fee estimate: {e}")
            return self.MIN_FEE_RATE

    def _cleanup_pool(self):
        """Clean up transaction pool"""
        try:
            current_time = time.time()
            
            # Skip if not enough time has passed
            if current_time - self.last_cleanup < self.cleanup_interval:
                return
            
            # Remove expired transactions
            expired_txs = []
            for txid, tx in self.transactions.items():
                if current_time > tx.expiration_time:
                    expired_txs.append(txid)
            
            for txid in expired_txs:
                self.remove_transaction(txid)
            
            self.last_cleanup = current_time
        except Exception as e:
            logging.error(f"Error cleaning up transaction pool: {e}")

    def get_transaction(self, txid):
        """Get transaction by ID"""
        return self.transactions.get(txid)

    def get_pool_size(self):
        """Get current pool size"""
        return len(self.transactions) 