import hashlib
import json
import logging

class UTXO:
    def __init__(self, txid, output_index, amount, address):
        self.txid = txid
        self.index = output_index  # Store as index for consistency
        self.amount = amount
        self.address = address
        self.spent = False
        self.spent_height = None  # Block height when UTXO was spent
        self.confirmation_depth = 0  # Number of confirmations since spending

    def to_dict(self):
        return {
            "txid": self.txid,
            "index": self.index,
            "amount": self.amount,
            "address": self.address,
            "spent": self.spent,
            "spent_height": self.spent_height,
            "confirmation_depth": self.confirmation_depth
        }

    @classmethod
    def from_dict(cls, data):
        utxo = cls(
            data["txid"],
            data.get("index", data.get("output_index")),  # Support both index and output_index
            data["amount"],
            data["address"]
        )
        utxo.spent = data["spent"]
        utxo.spent_height = data.get("spent_height")
        utxo.confirmation_depth = data.get("confirmation_depth", 0)
        return utxo

    def get_id(self):
        """Get unique identifier for this UTXO"""
        return f"{self.txid}:{self.index}"

class BTreeNode:
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.keys = []  # Store keys as strings
        self.values = []  # Store UTXO objects
        self.children = []  # Child nodes
        self.parent = None

    def to_dict(self):
        """Convert node to dictionary format"""
        return {
            'is_leaf': self.is_leaf,
            'keys': [str(key) for key in self.keys],
            'values': [utxo.to_dict() for utxo in self.values],
            'children': [child.to_dict() for child in self.children] if not self.is_leaf else []
        }

    @classmethod
    def from_dict(cls, data):
        """Create node from dictionary format"""
        node = cls(is_leaf=data['is_leaf'])
        node.keys = [str(key) for key in data['keys']]
        node.values = [UTXO.from_dict(utxo_dict) for utxo_dict in data['values']]
        if not node.is_leaf:
            node.children = [cls.from_dict(child_dict) for child_dict in data['children']]
            for child in node.children:
                child.parent = node
        return node

class UTXOBTree:
    def __init__(self, degree=3):
        self.root = BTreeNode(is_leaf=True)
        self.degree = degree  # Minimum degree of the B-tree
        self.size = 0  # Number of UTXOs stored

    def _normalize_key(self, key):
        """Ensure key is a string for comparison"""
        if isinstance(key, UTXO):
            return key.get_id()
        return str(key)

    def to_dict(self):
        """Convert B-tree to dictionary format for serialization"""
        return self.root.to_dict()

    @classmethod
    def from_dict(cls, data, degree=3):
        """Create B-tree from dictionary format"""
        tree = cls(degree)
        tree.root = BTreeNode.from_dict(data)
        return tree

    def search(self, utxo_id):
        """Search for a UTXO by ID"""
        return self._search_node(self.root, self._normalize_key(utxo_id))

    def _search_node(self, node, utxo_id):
        """Search for a UTXO in a specific node"""
        i = 0
        while i < len(node.keys) and utxo_id > self._normalize_key(node.keys[i]):
            i += 1

        if i < len(node.keys) and utxo_id == self._normalize_key(node.keys[i]):
            return node.values[i]

        if node.is_leaf:
            return None

        return self._search_node(node.children[i], utxo_id)

    def insert(self, utxo_id, utxo):
        """Insert a new UTXO"""
        root = self.root
        normalized_id = self._normalize_key(utxo_id)

        if len(root.keys) == (2 * self.degree) - 1:
            new_root = BTreeNode(is_leaf=False)
            new_root.children.append(self.root)
            self.root.parent = new_root
            self.root = new_root
            self._split_child(new_root, 0)
            self._insert_non_full(new_root, normalized_id, utxo)
        else:
            self._insert_non_full(root, normalized_id, utxo)

        self.size += 1

    def _insert_non_full(self, node, utxo_id, utxo):
        """Insert a UTXO into a non-full node"""
        i = len(node.keys) - 1

        if node.is_leaf:
            while i >= 0 and utxo_id < self._normalize_key(node.keys[i]):
                i -= 1
            node.keys.insert(i + 1, utxo_id)
            node.values.insert(i + 1, utxo)
        else:
            while i >= 0 and utxo_id < self._normalize_key(node.keys[i]):
                i -= 1
            i += 1

            if len(node.children[i].keys) == (2 * self.degree) - 1:
                self._split_child(node, i)
                if utxo_id > self._normalize_key(node.keys[i]):
                    i += 1

            self._insert_non_full(node.children[i], utxo_id, utxo)

    def _split_child(self, parent, index):
        """Split a child node"""
        degree = self.degree
        child = parent.children[index]
        new_node = BTreeNode(is_leaf=child.is_leaf)
        new_node.parent = parent

        parent.keys.insert(index, child.keys[degree - 1])
        parent.values.insert(index, child.values[degree - 1])
        parent.children.insert(index + 1, new_node)

        new_node.keys = child.keys[degree:(2 * degree) - 1]
        new_node.values = child.values[degree:(2 * degree) - 1]
        child.keys = child.keys[0:degree - 1]
        child.values = child.values[0:degree - 1]

        if not child.is_leaf:
            new_node.children = child.children[degree:2 * degree]
            child.children = child.children[0:degree]
            for c in new_node.children:
                c.parent = new_node

    def delete(self, utxo_id):
        """Delete a UTXO by ID"""
        if self.size == 0:
            return False

        normalized_id = self._normalize_key(utxo_id)
        if self._delete_from_node(self.root, normalized_id):
            self.size -= 1
            if len(self.root.keys) == 0 and not self.root.is_leaf:
                self.root = self.root.children[0]
                self.root.parent = None
            return True
        return False

    def _delete_from_node(self, node, utxo_id):
        """Delete a UTXO from a specific node"""
        i = 0
        while i < len(node.keys) and utxo_id > self._normalize_key(node.keys[i]):
            i += 1

        if i < len(node.keys) and utxo_id == self._normalize_key(node.keys[i]):
            if node.is_leaf:
                node.keys.pop(i)
                node.values.pop(i)
                return True
            else:
                return self._delete_from_internal_node(node, i)
        elif not node.is_leaf:
            return self._delete_from_node(node.children[i], utxo_id)
        return False

    def _delete_from_internal_node(self, node, index):
        """Delete from an internal node"""
        if len(node.children[index].keys) >= self.degree:
            predecessor = self._get_predecessor(node, index)
            node.keys[index] = predecessor.keys[-1]
            node.values[index] = predecessor.values[-1]
            return self._delete_from_node(node.children[index], predecessor.keys[-1])
        elif len(node.children[index + 1].keys) >= self.degree:
            successor = self._get_successor(node, index)
            node.keys[index] = successor.keys[0]
            node.values[index] = successor.values[0]
            return self._delete_from_node(node.children[index + 1], successor.keys[0])
        else:
            return self._merge_nodes(node, index)

    def _get_predecessor(self, node, index):
        """Get the predecessor of a key"""
        current = node.children[index]
        while not current.is_leaf:
            current = current.children[-1]
        return current

    def _get_successor(self, node, index):
        """Get the successor of a key"""
        current = node.children[index + 1]
        while not current.is_leaf:
            current = current.children[0]
        return current

    def _merge_nodes(self, node, index):
        """Merge two nodes"""
        child = node.children[index]
        sibling = node.children[index + 1]

        child.keys.append(node.keys[index])
        child.values.append(node.values[index])
        child.keys.extend(sibling.keys)
        child.values.extend(sibling.values)

        if not child.is_leaf:
            child.children.extend(sibling.children)
            for c in sibling.children:
                c.parent = child

        node.keys.pop(index)
        node.values.pop(index)
        node.children.pop(index + 1)

        return self._delete_from_node(child, child.keys[-1])

    def range_query(self, start_id, end_id):
        """Get all UTXOs in a range"""
        results = []
        self._range_query_node(self.root, self._normalize_key(start_id), self._normalize_key(end_id), results)
        return results

    def _range_query_node(self, node, start_id, end_id, results):
        """Get UTXOs in a range from a specific node"""
        i = 0
        while i < len(node.keys) and start_id > self._normalize_key(node.keys[i]):
            i += 1

        if not node.is_leaf:
            self._range_query_node(node.children[i], start_id, end_id, results)

        while i < len(node.keys) and self._normalize_key(node.keys[i]) <= end_id:
            results.append(node.values[i])
            if not node.is_leaf:
                self._range_query_node(node.children[i + 1], start_id, end_id, results)
            i += 1

class UTXOSet:
    def __init__(self):
        self.utxo_tree = UTXOBTree()  # B-tree for UTXO storage
        self.address_utxos = {}  # Map of addresses to their UTXOs
        self.merkle_root = None  # Merkle root of UTXO set
        self.merkle_tree = {}  # Merkle tree for UTXO set
        self.last_commitment_height = 0  # Height of last commitment
        self.pruning_enabled = True  # Whether UTXO pruning is enabled
        self.min_confirmations = 100  # Minimum confirmations before pruning
        self.last_prune_height = 0  # Height of last prune operation
        self.snapshots = {}  # Map of block heights to UTXO set snapshots
        self.snapshot_interval = 1000  # Take snapshots every 1000 blocks
        self.max_snapshots = 10  # Maximum number of snapshots to keep

    def __len__(self):
        """Return the number of unspent UTXOs in the set"""
        return len([utxo for utxo in self.utxo_tree.range_query("", "z") if not utxo.spent])

    def add_utxo(self, utxo):
        """Add a new UTXO to the set"""
        utxo_id = utxo.get_id()
        self.utxo_tree.insert(utxo_id, utxo)
        
        # Add to address index
        if utxo.address not in self.address_utxos:
            self.address_utxos[utxo.address] = []
        self.address_utxos[utxo.address].append(utxo_id)
        
        # Update merkle tree
        self._update_merkle_tree()

    def get_utxo(self, utxo_id):
        """Get a UTXO by its ID"""
        return self.utxo_tree.search(utxo_id)

    def spend_utxo(self, utxo_id, current_height):
        """Mark a UTXO as spent"""
        utxo = self.get_utxo(utxo_id)
        if utxo:
            utxo.spent = True
            utxo.spent_height = current_height
            
            # Remove from address index
            if utxo.address in self.address_utxos:
                self.address_utxos[utxo.address].remove(utxo_id)
                if not self.address_utxos[utxo.address]:
                    del self.address_utxos[utxo.address]
            
            # Update merkle tree
            self._update_merkle_tree()

    def _update_merkle_tree(self):
        """Update the merkle tree for the UTXO set"""
        # Get all unspent UTXOs
        unspent_utxos = [utxo for utxo in self.utxo_tree.range_query("", "z") if not utxo.spent]
        
        # Sort UTXOs by ID for deterministic ordering
        unspent_utxos.sort(key=lambda x: x.get_id())
        
        # Create leaf nodes
        leaves = [self._hash_utxo(utxo) for utxo in unspent_utxos]
        
        # Build merkle tree
        self.merkle_tree = self._build_merkle_tree(leaves)
        
        # Update merkle root
        if self.merkle_tree:
            self.merkle_root = self.merkle_tree[0]

    def _hash_utxo(self, utxo):
        """Calculate hash of a UTXO"""
        utxo_data = f"{utxo.txid}{utxo.index}{utxo.amount}{utxo.address}"
        return hashlib.sha256(utxo_data.encode()).hexdigest()

    def _build_merkle_tree(self, leaves):
        """Build a merkle tree from leaves"""
        if not leaves:
            return []
            
        tree = [leaves]
        current_level = leaves
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    combined = current_level[i] + current_level[i + 1]
                else:
                    combined = current_level[i] + current_level[i]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            tree.append(next_level)
            current_level = next_level
            
        return tree

    def get_merkle_proof(self, utxo_id):
        """Get merkle proof for a UTXO"""
        if utxo_id not in self.utxo_tree.range_query("", "z"):
            return None
            
        utxo = self.get_utxo(utxo_id)
        if utxo.spent:
            return None
            
        # Find the index of the UTXO in the leaves
        unspent_utxos = [utxo for utxo in self.utxo_tree.range_query("", "z") if not utxo.spent]
        unspent_utxos.sort(key=lambda x: x.get_id())
        
        try:
            index = unspent_utxos.index(utxo)
        except ValueError:
            return None
            
        # Build the proof
        proof = []
        current_level = self.merkle_tree[0]
        
        while len(current_level) > 1:
            if index % 2 == 1:
                proof.append(current_level[index - 1])
            else:
                if index + 1 < len(current_level):
                    proof.append(current_level[index + 1])
                else:
                    proof.append(current_level[index])
            index = index // 2
            current_level = self.merkle_tree[len(proof)]
            
        return proof

    def verify_merkle_proof(self, utxo, proof, merkle_root):
        """Verify a merkle proof for a UTXO"""
        if not proof:
            return False
            
        # Calculate hash of the UTXO
        current_hash = self._hash_utxo(utxo)
        
        # Verify the proof
        for sibling_hash in proof:
            if current_hash < sibling_hash:
                combined = current_hash + sibling_hash
            else:
                combined = sibling_hash + current_hash
            current_hash = hashlib.sha256(combined.encode()).hexdigest()
            
        return current_hash == merkle_root

    def commit(self, block_height):
        """Create a commitment for the UTXO set at a specific block height"""
        if block_height <= self.last_commitment_height:
            return None
            
        self.last_commitment_height = block_height
        return {
            "merkle_root": self.merkle_root,
            "block_height": block_height,
            "utxo_count": len([utxo for utxo in self.utxo_tree.range_query("", "z") if not utxo.spent])
        }

    def get_address_balance(self, address):
        """Get the balance for an address"""
        balance = 0
        for utxo in self.get_address_utxos(address):
            balance += utxo.amount
        return balance

    def get_address_utxos(self, address):
        """Get all unspent UTXOs for an address"""
        utxos = []
        if address in self.address_utxos:
            for utxo_id in self.address_utxos[address]:
                utxo = self.get_utxo(utxo_id)
                if utxo and not utxo.spent:
                    utxos.append(utxo)
        return utxos

    def to_dict(self):
        """Convert UTXO set to dictionary format"""
        try:
            # Convert UTXO tree to dictionary format
            utxo_tree_dict = self.utxo_tree.to_dict()
            
            # Convert address_utxos to use string keys
            address_utxos_dict = {}
            for address, utxo_ids in self.address_utxos.items():
                address_utxos_dict[str(address)] = [str(utxo_id) for utxo_id in utxo_ids]
                
            # Convert snapshots to use string keys
            snapshots_dict = {}
            for height, snapshot in self.snapshots.items():
                snapshots_dict[str(height)] = {
                    "utxos": {str(k): v for k, v in snapshot["utxos"].items()},
                    "address_utxos": {str(k): [str(v) for v in vs] for k, vs in snapshot["address_utxos"].items()},
                    "merkle_root": str(snapshot["merkle_root"]) if snapshot["merkle_root"] else None,
                    "merkle_tree": snapshot["merkle_tree"],  # merkle_tree is a list, not a dict
                    "block_height": snapshot["block_height"]
                }
                
            return {
                "utxo_tree": utxo_tree_dict,
                "address_utxos": address_utxos_dict,
                "merkle_root": str(self.merkle_root) if self.merkle_root else None,
                "merkle_tree": self.merkle_tree,  # merkle_tree is a list, not a dict
                "last_commitment_height": self.last_commitment_height,
                "pruning_enabled": self.pruning_enabled,
                "min_confirmations": self.min_confirmations,
                "last_prune_height": self.last_prune_height,
                "snapshots": snapshots_dict,
                "snapshot_interval": self.snapshot_interval,
                "max_snapshots": self.max_snapshots
            }
        except Exception as e:
            logging.error(f"Error converting UTXO set to dictionary: {e}")
            return {}

    @classmethod
    def from_dict(cls, data):
        """Create UTXO set from dictionary format"""
        utxo_set = cls()
        utxo_set.utxo_tree = UTXOBTree.from_dict(data["utxo_tree"])
        utxo_set.address_utxos = {address: [utxo_id for utxo_id in utxo_ids] for address, utxo_ids in data["address_utxos"].items()}
        utxo_set.merkle_root = data["merkle_root"]
        utxo_set.merkle_tree = data["merkle_tree"]
        utxo_set.last_commitment_height = data["last_commitment_height"]
        utxo_set.pruning_enabled = data["pruning_enabled"]
        utxo_set.min_confirmations = data["min_confirmations"]
        utxo_set.last_prune_height = data["last_prune_height"]
        utxo_set.snapshots = {int(height): snapshot for height, snapshot in data["snapshots"].items()}
        utxo_set.snapshot_interval = data["snapshot_interval"]
        utxo_set.max_snapshots = data["max_snapshots"]
        return utxo_set

    def save_to_file(self, filename):
        """Save UTXO set to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.to_dict(), f)
        except Exception as e:
            logging.error(f"Error saving UTXO set: {e}")

    @classmethod
    def load_from_file(cls, filename):
        """Load UTXO set from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                return cls.from_dict(data)
        except FileNotFoundError:
            return cls()
        except Exception as e:
            logging.error(f"Error loading UTXO set: {e}")
            return cls()

    def update_from_dict(self, data):
        """Update UTXO set from dictionary format"""
        try:
            # Clear current state
            self.utxo_tree = UTXOBTree()
            self.address_utxos.clear()
            
            # Load new state
            for utxo_id, utxo_data in data["utxos"].items():
                utxo = UTXO.from_dict(utxo_data)
                self.utxo_tree.insert(utxo_id, utxo)
            
            self.address_utxos = data["address_utxos"]
            self.merkle_root = data["merkle_root"]
            self.merkle_tree = data["merkle_tree"]
            self.last_commitment_height = data["last_commitment_height"]
            logging.info("UTXO set updated successfully")
        except Exception as e:
            logging.error(f"Error updating UTXO set: {e}")

    def update(self, block):
        """Update UTXO set with transactions from a block"""
        try:
            # Convert block to dict if needed
            block_dict = block.to_dict() if hasattr(block, 'to_dict') else block
            
            # Process each transaction in the block
            for tx in block_dict['transactions']:
                # Handle coinbase and genesis transactions
                if tx['sender_address'] in ["COINBASE", "GENESIS"]:
                    # Add output to UTXO set
                    for i, output in enumerate(tx['outputs']):
                        utxo = UTXO(
                            txid=tx['txid'],
                            output_index=i,
                            amount=output['amount'],
                            address=output['address']
                        )
                        self.add_utxo(utxo)
                    continue

                # Remove spent inputs from UTXO set if present
                if 'inputs' in tx:
                    for input_tx in tx['inputs']:
                        utxo_id = f"{input_tx['txid']}:{input_tx['output_index']}"
                        utxo = self.get_utxo(utxo_id)
                        if not utxo:
                            logging.error(f"Input UTXO not found: {utxo_id}")
                            return False
                        if utxo.spent:
                            logging.error(f"Input UTXO already spent: {utxo_id}")
                            return False
                        self.spend_utxo(utxo_id, block_height)

                # Add new outputs to UTXO set
                for i, output in enumerate(tx['outputs']):
                    utxo = UTXO(
                        txid=tx['txid'],
                        output_index=i,
                        amount=output['amount'],
                        address=output['address']
                    )
                    self.add_utxo(utxo)

            return True
        except Exception as e:
            logging.error(f"Error updating UTXO set: {e}")
            return False

    def update_confirmation_depths(self, current_height):
        """Update confirmation depths for all spent UTXOs"""
        for utxo in self.utxo_tree.range_query("", "z"):
            if utxo.spent and utxo.spent_height is not None:
                utxo.confirmation_depth = current_height - utxo.spent_height

    def prune_utxos(self, current_height):
        """Prune UTXOs that have been spent and have sufficient confirmations"""
        if not self.pruning_enabled:
            return

        # Update confirmation depths
        self.update_confirmation_depths(current_height)

        # Find UTXOs to prune
        utxos_to_prune = []
        total_utxos = 0
        spent_utxos = 0
        confirmed_spent = 0

        # Get all UTXOs
        for utxo_id in self.utxo_tree.range_query("", "z"):
            total_utxos += 1
            utxo = self.get_utxo(utxo_id)
            if utxo:
                if utxo.spent:
                    spent_utxos += 1
                    if utxo.confirmation_depth >= self.min_confirmations:
                        confirmed_spent += 1
                        utxos_to_prune.append(utxo_id)

        # Log pruning statistics
        logging.info(f"UTXO pruning stats at height {current_height}:")
        logging.info(f"Total UTXOs: {total_utxos}")
        logging.info(f"Spent UTXOs: {spent_utxos}")
        logging.info(f"Spent UTXOs with sufficient confirmations: {confirmed_spent}")
        logging.info(f"UTXOs to prune: {len(utxos_to_prune)}")

        # Remove pruned UTXOs
        for utxo_id in utxos_to_prune:
            self.utxo_tree.delete(utxo_id)

        self.last_prune_height = current_height
        logging.info(f"Pruned {len(utxos_to_prune)} UTXOs at height {current_height}")

    def get_pruning_stats(self):
        """Get statistics about UTXO pruning"""
        total_utxos = len(self.utxo_tree.range_query("", "z"))
        spent_utxos = sum(1 for utxo in self.utxo_tree.range_query("", "z") if utxo.spent)
        unspent_utxos = total_utxos - spent_utxos
        prunable_utxos = sum(1 for utxo in self.utxo_tree.range_query("", "z") 
                            if utxo.spent and utxo.confirmation_depth >= self.min_confirmations)

        return {
            "total_utxos": total_utxos,
            "spent_utxos": spent_utxos,
            "unspent_utxos": unspent_utxos,
            "prunable_utxos": prunable_utxos,
            "pruning_enabled": self.pruning_enabled,
            "min_confirmations": self.min_confirmations,
            "last_prune_height": self.last_prune_height
        }

    def create_snapshot(self, block_height):
        """Create a snapshot of the UTXO set at a specific block height"""
        try:
            # Only create snapshots at specified intervals
            if block_height % self.snapshot_interval != 0:
                return

            # Create snapshot
            snapshot = {
                "utxos": {},
                "address_utxos": self.address_utxos,
                "merkle_root": self.merkle_root,
                "merkle_tree": self.merkle_tree,
                "block_height": block_height
            }

            # Convert UTXOs to dictionary format
            for utxo_id in self.utxo_tree.range_query("", "z"):
                utxo = self.utxo_tree.search(utxo_id)
                if utxo:
                    snapshot["utxos"][utxo_id] = utxo.to_dict()

            # Store snapshot
            self.snapshots[block_height] = snapshot

            # Cleanup old snapshots
            self._cleanup_snapshots()

            logging.info(f"Created UTXO set snapshot at height {block_height}")
            return snapshot
        except Exception as e:
            logging.error(f"Error creating UTXO set snapshot: {e}")
            return None

    def _cleanup_snapshots(self):
        """Remove old snapshots to maintain maximum snapshot count"""
        try:
            # Sort snapshots by height
            sorted_heights = sorted(self.snapshots.keys())
            
            # Remove oldest snapshots if we exceed the maximum
            while len(sorted_heights) > self.max_snapshots:
                oldest_height = sorted_heights.pop(0)
                del self.snapshots[oldest_height]
                logging.info(f"Removed old UTXO set snapshot at height {oldest_height}")
        except Exception as e:
            logging.error(f"Error cleaning up UTXO set snapshots: {e}")

    def restore_from_snapshot(self, block_height):
        """Restore UTXO set from a snapshot at a specific block height"""
        try:
            # Find the closest snapshot
            closest_height = None
            for height in sorted(self.snapshots.keys()):
                if height <= block_height:
                    closest_height = height
                else:
                    break

            if closest_height is None:
                logging.error(f"No suitable snapshot found for height {block_height}")
                return False

            # Get the snapshot
            snapshot = self.snapshots[closest_height]
            
            # Clear current state
            self.utxo_tree = UTXOBTree()
            self.address_utxos.clear()
            
            # Restore UTXOs
            for utxo_id, utxo_data in snapshot["utxos"].items():
                utxo = UTXO.from_dict(utxo_data)
                self.utxo_tree.insert(utxo_id, utxo)
                
                # Restore address index
                if utxo.address not in self.address_utxos:
                    self.address_utxos[utxo.address] = []
                self.address_utxos[utxo.address].append(utxo_id)

            # Restore merkle tree
            self.merkle_root = snapshot["merkle_root"]
            self.merkle_tree = snapshot["merkle_tree"]
            
            logging.info(f"Restored UTXO set from snapshot at height {closest_height}")
            return True
        except Exception as e:
            logging.error(f"Error restoring UTXO set from snapshot: {e}")
            return False

    def get_snapshot_info(self):
        """Get information about available snapshots"""
        return {
            "snapshot_count": len(self.snapshots),
            "snapshot_heights": sorted(self.snapshots.keys()),
            "snapshot_interval": self.snapshot_interval,
            "max_snapshots": self.max_snapshots
        } 