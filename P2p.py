import socket
import threading
import json
import time
from Block import Block
import logging
import random

class P2PNetwork:
    def __init__(self, host, port, blockchain, message_queue, bootstrap_nodes=None):
        self.host = host
        self.port = port
        self.blockchain = blockchain
        self.message_queue = message_queue
        self.peers = {}  # Map of (host, port) to socket connections
        self.server = None
        self.bootstrap_nodes = bootstrap_nodes or []
        self.peer_lock = threading.RLock()  # Reentrant lock for peer operations
        self.blockchain_lock = threading.Lock()  # Lock for blockchain operations
        self.message_queue_lock = threading.Lock()  # Lock for message queue operations
        self.connected = False
        self.running = True
        
        # Initialize miner
        from Miner import Miner
        self.blockchain.miner = Miner(blockchain, None)  # No wallet needed for mining

    def start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        logging.info(f"🌐 P2P Server started on {self.host}:{self.port}")
        
        # Start bootstrap connection thread
        threading.Thread(target=self.connect_to_bootstrap_nodes, daemon=True).start()
        
        # Start peer maintenance thread
        threading.Thread(target=self.maintain_peers, daemon=True).start()
        
        while self.running:
            try:
                client, addr = self.server.accept()
                threading.Thread(target=self.handle_client, args=(client, addr), daemon=True).start()
            except Exception as e:
                if self.running:  # Only log if not shutting down
                    logging.error(f"Server error: {e}")
                break

    def stop_server(self):
        """Gracefully stop the server and close all connections"""
        self.running = False
        if self.server:
            self.server.close()
        
        with self.peer_lock:
            for sock in self.peers.values():
                try:
                    sock.close()
                except:
                    pass
            self.peers.clear()

    def maintain_peers(self):
        """Maintain peer connections and attempt reconnection if needed"""
        while self.running:
            try:
                with self.peer_lock:
                    # Check each peer connection
                    for peer, sock in list(self.peers.items()):
                        try:
                            # Try to send a ping message
                            sock.send(json.dumps({"type": "ping"}).encode('utf-8') + b"\n")
                        except:
                            # Connection is dead, remove it
                            logging.debug(f"Lost connection to peer {peer}")
                            sock.close()
                            del self.peers[peer]
                            
                    # Try to reconnect to bootstrap nodes if we have no peers
                    if not self.peers:
                        self.connected = False
                        threading.Thread(target=self.connect_to_bootstrap_nodes, daemon=True).start()
                        
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logging.error(f"Error in peer maintenance: {e}")
                time.sleep(5)

    def connect_to_bootstrap_nodes(self):
        """Connect to bootstrap nodes and establish initial connections"""
        if self.connected:
            return
            
        for node_host, node_port in self.bootstrap_nodes:
            if (node_host, node_port) != (self.host, self.port):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    sock.connect((node_host, node_port))
                    logging.info(f"Connected to bootstrap node {node_host}:{node_port}")
                    
                    # Send hello message
                    hello_msg = {
                        "type": "hello",
                        "address": f"{self.host}:{self.port}"
                    }
                    sock.send(json.dumps(hello_msg).encode('utf-8') + b"\n")
                    
                    # Add to peers
                    with self.peer_lock:
                        self.peers[(node_host, node_port)] = sock
                    
                    # Start handler thread for this connection
                    threading.Thread(target=self.handle_client, args=(sock, (node_host, node_port)), daemon=True).start()
                    
                    # Request chain
                    with self.blockchain_lock:
                        request = {
                            "type": "request_chain",
                            "data": {
                                "height": len(self.blockchain.chain),
                                "work": self.blockchain.calculate_total_work(self.blockchain.chain)
                            }
                        }
                    sock.send(json.dumps(request).encode('utf-8') + b"\n")
                    
                    self.connected = True
                    break
                    
                except socket.error as e:
                    logging.error(f"Socket error connecting to {node_host}:{node_port}: {e}")
                except Exception as e:
                    logging.error(f"Error connecting to bootstrap node {node_host}:{node_port}: {e}")

    def handle_client(self, client, addr):
        """Handle client connection and messages"""
        logging.debug(f"🤝 New connection from {addr}")
        buffer = ""
        
        try:
            client.settimeout(10)  # Set socket timeout to 10 seconds
            while self.running:
                try:
                    data = client.recv(65536).decode('utf-8')
                    if not data:
                        break
                    
                    buffer += data
                    while "\n" in buffer:
                        message_str, buffer = buffer.split("\n", 1)
                        message = json.loads(message_str)
                        logging.debug(f"Processing message from {addr}: {message['type']}")
                        
                        if message["type"] == "hello":
                            peer_host, peer_port = message["address"].split(':')
                            peer_addr = (peer_host, int(peer_port))
                            with self.peer_lock:
                                if peer_addr not in self.peers:
                                    self.peers[peer_addr] = client
                            # Send hello response
                            response = {
                                "type": "hello_response",
                                "address": f"{self.host}:{self.port}"
                            }
                            client.send(json.dumps(response).encode('utf-8') + b"\n")
                            
                        elif message["type"] == "ping":
                            # Respond to ping with pong
                            client.send(json.dumps({"type": "pong"}).encode('utf-8') + b"\n")
                            
                        elif message["type"] == "get_balance":
                            # Handle get_balance request
                            try:
                                address = message.get("data", {}).get("address")
                                if not address:
                                    raise ValueError("Address not provided")
                                    
                                with self.blockchain_lock:
                                    balance = self.blockchain.get_balance(address)
                                response = {
                                    "type": "success",
                                    "data": {
                                        "address": address,
                                        "balance": balance
                                    }
                                }
                                client.send(json.dumps(response).encode('utf-8') + b"\n")
                                
                            except Exception as e:
                                logging.error(f"Error handling get_balance request: {e}")
                                error_response = {
                                    "type": "error",
                                    "error": str(e)
                                }
                                client.send(json.dumps(error_response).encode('utf-8') + b"\n")
                                
                        elif message["type"] == "get_utxos":
                            # Handle get_utxos request
                            try:
                                address = message.get("data", {}).get("address")
                                if not address:
                                    raise ValueError("Address not provided")
                                    
                                with self.blockchain_lock:
                                    utxos = self.blockchain.get_utxos(address)
                                response = {
                                    "type": "success",
                                    "data": {
                                        "address": address,
                                        "utxos": utxos
                                    }
                                }
                                client.send(json.dumps(response).encode('utf-8') + b"\n")
                                
                            except Exception as e:
                                logging.error(f"Error handling get_utxos request: {e}")
                                error_response = {
                                    "type": "error",
                                    "error": str(e)
                                }
                                client.send(json.dumps(error_response).encode('utf-8') + b"\n")
                            
                        elif message["type"] == "get_metrics":
                            # Handle get_metrics request
                            try:
                                # Get current chain length from last block index
                                chain_length = self.blockchain.chain[-1].index + 1 if self.blockchain.chain else 0
                                metrics = {
                                    "node": {
                                        "status": "online",
                                        "peers": len(self.peers),
                                        "active_peers": sum(1 for peer in self.peers.values() if not peer._closed)
                                    },
                                    "blockchain": {
                                        "chain_length": chain_length,
                                        "last_block": {
                                            "index": self.blockchain.chain[-1].index if self.blockchain.chain else None,
                                            "hash": self.blockchain.chain[-1].hash if self.blockchain.chain else None
                                        },
                                        "mining": {
                                            "status": "mining" if self.blockchain.transaction_pool else "idle",
                                            "transaction_pool_size": len(self.blockchain.transaction_pool)
                                        }
                                    }
                                }
                                response = {
                                    "type": "metrics",
                                    "data": metrics
                                }
                                client.send(json.dumps(response).encode('utf-8') + b"\n")
                            except Exception as e:
                                logging.error(f"Error handling get_metrics request: {e}")
                                error_response = {
                                    "type": "error",
                                    "error": str(e)
                                }
                                client.send(json.dumps(error_response).encode('utf-8') + b"\n")
                            
                        elif message["type"] == "add_transaction":
                            # Handle add_transaction request
                            try:
                                tx_data = message.get("data", {})
                                logging.info(f"Received transaction from {addr}: {tx_data}")
                                
                                # Add transaction to blockchain
                                with self.blockchain_lock:
                                    result = self.blockchain.add_transaction(tx_data)
                                
                                # Check result type
                                if result.get("type") == "success":
                                    logging.info(f"Added transaction to blockchain: {result.get('txid', 'unknown')}")
                                    
                                    # Broadcast transaction to other peers
                                    self.broadcast_transaction(tx_data, exclude_addr=addr)
                                    
                                    # Send success response
                                    response = {
                                        "type": "success",
                                        "data": {
                                            "txid": result.get("txid"),
                                            "status": "accepted"
                                        }
                                    }
                                else:
                                    # Send error response
                                    response = {
                                        "type": "error",
                                        "error": result.get("error", "Transaction rejected")
                                    }
                                
                                client.send(json.dumps(response).encode('utf-8') + b"\n")
                                
                            except Exception as e:
                                logging.error(f"Error handling add_transaction request: {e}")
                                error_response = {
                                    "type": "error",
                                    "error": str(e)
                                }
                                client.send(json.dumps(error_response).encode('utf-8') + b"\n")
                                
                        elif message["type"] == "mine_transaction":
                            # Handle mine_transaction request
                            try:
                                tx_data = message.get("data", {})
                                logging.info(f"Received mine_transaction request from {addr}")
                                
                                # Add transaction to blockchain if not already there
                                with self.blockchain_lock:
                                    result = self.blockchain.add_transaction(tx_data)
                                    
                                    if result.get("type") == "success":
                                        # Get transactions for block
                                        transactions = self.blockchain.get_transactions_for_block()
                                        
                                        # Mine block with transactions
                                        new_block = self.blockchain.miner.mine_block(transactions=transactions)
                                        
                                        if new_block:
                                            # Add block to blockchain
                                            if self.blockchain.add_block(new_block):
                                                response = {
                                                    "type": "success",
                                                    "data": {
                                                        "txid": result.get("txid"),
                                                        "block_hash": new_block.hash,
                                                        "status": "mined"
                                                    }
                                                }
                                            else:
                                                response = {
                                                    "type": "error",
                                                    "error": "Failed to add block to blockchain"
                                                }
                                        else:
                                            response = {
                                                "type": "error",
                                                "error": "Failed to mine block"
                                            }
                                    else:
                                        response = {
                                            "type": "error",
                                            "error": result.get("error", "Transaction rejected")
                                        }
                                
                                client.send(json.dumps(response).encode('utf-8') + b"\n")
                                
                            except Exception as e:
                                logging.error(f"Error handling mine_transaction request: {e}")
                                error_response = {
                                    "type": "error",
                                    "error": str(e)
                                }
                                client.send(json.dumps(error_response).encode('utf-8') + b"\n")
                            
                        elif message["type"] == "get_transactions":
                            # Handle get_transactions request
                            try:
                                bloom_filter = message.get("data", {}).get("bloom_filter_b64")
                                start_block = message.get("data", {}).get("since", 0)
                                
                                # Get transactions matching bloom filter
                                transactions = self.blockchain.get_transactions(bloom_filter_b64=bloom_filter, start_block=start_block)
                                
                                # Send response
                                response = {
                                    "type": "transactions",
                                    "transactions": transactions
                                }
                                client.send(json.dumps(response).encode('utf-8') + b"\n")
                                logging.info(f"Sent {len(transactions)} transactions to {addr}")
                                
                            except Exception as e:
                                logging.error(f"Error handling get_transactions request: {e}")
                                error_response = {
                                    "type": "error",
                                    "error": str(e)
                                }
                                client.send(json.dumps(error_response).encode('utf-8') + b"\n")
                            
                        elif message["type"] == "request_chain":
                            # Send our chain to the requesting peer in chunks
                            chain_data = {
                                "type": "chain",
                                "data": {
                                    "chain": [block.to_dict() for block in self.blockchain.chain],
                                    "utxo_set": self.blockchain.utxo_set.to_dict(),
                                    "used_txids": list(self.blockchain.used_txids),
                                    "transaction_pool": self.blockchain.transaction_pool
                                }
                            }
                            # Send in chunks of 65536 bytes
                            json_data = json.dumps(chain_data).encode('utf-8') + b"\n"
                            chunk_size = 65536
                            for i in range(0, len(json_data), chunk_size):
                                chunk = json_data[i:i + chunk_size]
                                client.send(chunk)
                            logging.info(f"Sent chain to {addr}")
                            
                        elif message["type"] == "chain":
                            try:
                                chain_data = message["data"]
                                received_chain = []
                                for b in chain_data["chain"]:
                                    block = Block(
                                        version=b.get("version", 1),
                                        previous_hash=b["previous_hash"],
                                        transactions=b["transactions"],
                                        difficulty_target=int(b["difficulty_target"], 16) if isinstance(b["difficulty_target"], str) else b["difficulty_target"],
                                        nonce=b["nonce"],
                                        timestamp=b["timestamp"],
                                        index=len(received_chain)
                                    )
                                    block.hash = b["hash"]
                                    received_chain.append(block)
                                
                                if received_chain:
                                    # Calculate total work
                                    received_work = self.blockchain.calculate_total_work(received_chain)
                                    current_work = self.blockchain.calculate_total_work(self.blockchain.chain)
                                    
                                    logging.info(f"Received chain from {addr} with length {len(received_chain)}, work: {received_work}")
                                    
                                    if received_work > current_work:
                                        if self.blockchain.validate_chain(received_chain):
                                            logging.info(f"Switching to better chain from {addr}")
                                            self.blockchain.chain = received_chain
                                            self.blockchain.block_hashes = {b.hash for b in received_chain}
                                            self.blockchain.utxo_set.update_from_dict(chain_data["utxo_set"])
                                            self.blockchain.used_txids = set(chain_data["used_txids"])
                                            self.blockchain.transaction_pool = chain_data["transaction_pool"]
                                            if self.blockchain.db_file:
                                                self.blockchain.save_to_db()
                                            logging.info(f"Switched to chain with more work: {received_work} > {current_work}")
                                            
                            except Exception as e:
                                logging.error(f"Error processing chain from {addr}: {e}")
                                
                        elif message["type"] == "block":
                            try:
                                block_data = message["data"]
                                block = Block(
                                    version=block_data.get("version", 1),
                                    previous_hash=block_data["previous_hash"],
                                    transactions=block_data["transactions"],
                                    difficulty_target=int(block_data["difficulty_target"], 16) if isinstance(block_data["difficulty_target"], str) else block_data["difficulty_target"],
                                    nonce=block_data["nonce"],
                                    timestamp=block_data["timestamp"]
                                )
                                block.hash = block_data["hash"]
                                block.index = block_data["index"]
                                
                                if self.blockchain.add_block(block):
                                    logging.info(f"✅ Received valid block {block.hash[:8]} from {addr}")
                                    # Broadcast to other peers
                                    self.broadcast_block(block.to_dict(), exclude=addr)
                                else:
                                    logging.warning(f"❌ Received invalid block from {addr}")
                                    
                            except Exception as e:
                                logging.error(f"Error processing block from {addr}: {e}")
                                
                        elif message["type"] == "get_status":
                            # Handle get_status request
                            try:
                                logging.info(f"Handling get_status request from {addr}")
                                # Get block size information
                                avg_block_size = 0
                                if len(self.blockchain.chain) > 0:
                                    total_size = sum(self.blockchain.calculate_block_size(block) for block in self.blockchain.chain[-100:])
                                    avg_block_size = total_size / min(100, len(self.blockchain.chain))

                                status = {
                                    "height": len(self.blockchain.chain),
                                    "avg_block_size": avg_block_size,
                                    "block_size_limit": self.blockchain.MAX_BLOCK_SIZE,
                                    "latest_block": self.blockchain.chain[-1].to_dict() if self.blockchain.chain else None
                                }
                                response = {
                                    "type": "success",
                                    "status": status
                                }
                                logging.info(f"Sending status response to {addr}: {json.dumps(response)}")
                                client.send(json.dumps(response).encode('utf-8') + b"\n")
                                logging.info(f"Status response sent to {addr}")
                                
                            except Exception as e:
                                logging.error(f"Error handling get_status request: {e}")
                                error_response = {
                                    "type": "error",
                                    "error": str(e)
                                }
                                client.send(json.dumps(error_response).encode('utf-8') + b"\n")
                            
                        elif message["type"] == "get_transaction_history":
                            # Handle get_transaction_history request
                            try:
                                address = message.get("data", {}).get("address")
                                if not address:
                                    raise ValueError("Address not provided")
                                    
                                with self.blockchain_lock:
                                    transactions = self.blockchain.get_transaction_history(address)
                                response = {
                                    "type": "success",
                                    "data": {
                                        "address": address,
                                        "transactions": transactions
                                    }
                                }
                                client.send(json.dumps(response).encode('utf-8') + b"\n")
                                
                            except Exception as e:
                                logging.error(f"Error handling get_transaction_history request: {e}")
                                error_response = {
                                    "type": "error",
                                    "error": str(e)
                                }
                                client.send(json.dumps(error_response).encode('utf-8') + b"\n")
                            
                except socket.timeout:
                    logging.debug(f"Socket timeout for {addr}, sending ping")
                    try:
                        client.send(json.dumps({"type": "ping"}).encode('utf-8') + b"\n")
                    except:
                        break
                except Exception as e:
                    logging.debug(f"Error handling client {addr}: {e}")
                    break
            
        except Exception as e:
            logging.debug(f"Error handling client {addr}: {e}")
            
        finally:
            with self.peer_lock:
                if addr in self.peers:
                    del self.peers[addr]
            try:
                client.close()
            except:
                pass
            logging.debug(f"Connection closed with {addr}")

    def send_message(self, peer, message):
        """Send a message to a peer"""
        try:
            sock = self.peers.get(peer)
            if sock:
                sock.send(json.dumps(message).encode('utf-8') + b"\n")
            else:
                logging.error(f"No socket found for peer {peer}")
                if peer in self.peers:
                    del self.peers[peer]
        except Exception as e:
            logging.debug(f"Error sending message to {peer}: {e}")
            if peer in self.peers:
                del self.peers[peer]

    def broadcast_block(self, block_data, exclude=None):
        """Broadcast a block to all peers except the excluded one"""
        message = {
            "type": "block",
            "data": block_data
        }
        peers_to_remove = []
        for peer in self.peers:
            if exclude and peer == exclude:
                continue
            try:
                self.send_message(peer, message)
            except Exception as e:
                logging.error(f"Error broadcasting block to {peer}: {e}")
                peers_to_remove.append(peer)
        
        # Remove failed peers
        for peer in peers_to_remove:
            if peer in self.peers:
                del self.peers[peer]

    def broadcast_transaction(self, tx_data, exclude_addr=None):
        """Broadcast transaction to all peers except the sender"""
        message = {
            "type": "add_transaction",
            "data": tx_data
        }
        self.broadcast_message(message, exclude_addr)

    def broadcast_message(self, message, exclude_addr=None):
        """Broadcast a message to all connected peers except the excluded address"""
        with self.peer_lock:
            for peer_addr, sock in list(self.peers.items()):
                if peer_addr != exclude_addr:
                    try:
                        sock.send(json.dumps(message).encode('utf-8') + b"\n")
                    except Exception as e:
                        logging.error(f"Error broadcasting to {peer_addr}: {e}")
                        # Connection is dead, remove it
                        sock.close()
                        del self.peers[peer_addr]

    def sync_chain(self):
        """Synchronize chain with peers"""
        try:
            if not self.peers:
                logging.info("No peers available for chain sync")
                return False

            # Get current chain state
            current_work = self.blockchain.calculate_total_work(self.blockchain.chain)
            current_height = len(self.blockchain.chain)
            
            # Request chain from all peers
            for peer in self.peers:
                try:
                    host, port = peer
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    sock.connect((host, port))
                    
                    # Request chain
                    request = {
                        "type": "request_chain",
                        "data": {
                            "height": current_height,
                            "work": current_work
                        }
                    }
                    sock.send(json.dumps(request).encode('utf-8') + b"\n")
                    
                    # Receive response
                    response = sock.recv(65536).decode('utf-8')
                    chain_data = json.loads(response)
                    sock.close()
                    
                    if not chain_data or 'data' not in chain_data or 'chain' not in chain_data['data']:
                        logging.debug(f"Invalid chain data from {host}:{port}")
                        continue
                    
                    # Validate received chain
                    received_chain = []
                    for i, b in enumerate(chain_data['data']['chain']):
                        # Convert block data to Block object
                        block = Block(
                            version=b.get('version', 1),
                            previous_hash=b['previous_hash'],
                            transactions=b['transactions'],
                            difficulty_target=int(b['difficulty_target'], 16) if isinstance(b['difficulty_target'], str) else b['difficulty_target'],
                            nonce=b['nonce'],
                            timestamp=b['timestamp']
                        )
                        block.hash = b['hash']
                        block.index = i  # Set the block index
                        received_chain.append(block)
                    
                    if not received_chain:
                        logging.debug(f"Empty chain received from {host}:{port}")
                        continue
                    
                    # Calculate total work of received chain
                    received_work = self.blockchain.calculate_total_work(received_chain)
                    logging.info(f"Received chain from {host}:{port} with length {len(received_chain)}, work: {received_work}")
                    
                    # Only switch chains if the received chain has more work
                    if received_work > current_work:
                        if self.blockchain.validate_chain(received_chain):
                            logging.info(f"Switching to better chain from {host}:{port}")
                            self.blockchain.chain = received_chain
                            self.blockchain.block_hashes = {b.hash for b in received_chain}
                            self.blockchain.utxo_set.update_from_dict(chain_data['data']['utxo_set'])
                            self.blockchain.used_txids = set(chain_data['data']['used_txids'])
                            self.blockchain.transaction_pool = chain_data['data']['transaction_pool']
                            if self.blockchain.db_file:
                                self.blockchain.save_to_db()
                            logging.info(f"Switched to chain with more work: {received_work} > {current_work}")
                            return True
                        else:
                            logging.warning(f"Received invalid chain from {host}:{port}")
                    
                except Exception as e:
                    logging.error(f"Error syncing chain with {peer}: {e}")
                    continue
                    
            return False
        except Exception as e:
            logging.error(f"Error in sync_chain: {e}")
            return False

    def calculate_total_work(self, chain):
        total_work = 0
        for block in chain:
            # Handle both Block objects and dictionaries
            if isinstance(block, dict):
                target = block.get('difficulty_target', 0)
            else:
                target = block.difficulty_target
            
            # Convert target to integer if it's a string
            if isinstance(target, str):
                try:
                    target = int(target, 16) if target.startswith('0x') else int(target)
                except ValueError:
                    target = 0
            
            work = 2**256 // (target + 1)
            total_work += work
        return total_work

    def validate_chain(self, chain):
        if not chain:
            return False
        
        # Check genesis block
        if isinstance(chain[0], dict):
            if chain[0].get('previous_hash') != "0" * 64:
                return False
        else:
            if chain[0].previous_hash != "0" * 64:
                return False
        
        # Check each block
        for i in range(1, len(chain)):
            if not self.blockchain.is_valid_block(chain[i], chain[i-1]):
                return False
        
        return True
