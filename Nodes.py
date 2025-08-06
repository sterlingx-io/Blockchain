import logging

class Node:
    def get_metrics(self):
        """Get node metrics"""
        try:
            # Get current mining status
            mining_status = "idle"
            if hasattr(self, 'miner') and self.miner:
                # Check if there are transactions in the pool that could be mined
                if self.blockchain.transaction_pool:
                    mining_status = "mining"
            
            # Get blockchain metrics
            chain_length = len(self.blockchain.chain)
            last_block = self.blockchain.chain[-1] if chain_length > 0 else None
            
            # Calculate network metrics
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
                        "transaction_pool_size": len(self.blockchain.transaction_pool)
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