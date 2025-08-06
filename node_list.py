#!/usr/bin/env python3
import json
import os
import logging
import time
import socket
import threading
import sys
import signal
from typing import Dict, List, Tuple
from multiprocessing import Process, Queue, Event

class NodeList:
    def __init__(self, file_path: str = None):
        """Initialize the node list manager"""
        if file_path is None:
            # Use default path in user's home directory
            home_dir = os.path.expanduser("~")
            self.file_path = os.path.join(home_dir, ".local", "share", "sterlingx", "node_list.json")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        else:
            self.file_path = file_path
        self.nodes: Dict[str, Dict] = {}
        self.load_nodes()
        self.scan_thread = None
        self.scan_active = False
        self.update_queue = Queue()
        self.stop_event = Event()

    def load_nodes(self) -> None:
        """Load nodes from the JSON file"""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    self.nodes = json.load(f)
            else:
                self.nodes = {}
                self.save_nodes()
        except Exception as e:
            logging.error(f"Error loading node list: {e}")
            self.nodes = {}

    def save_nodes(self) -> None:
        """Save nodes to the JSON file"""
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.nodes, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving node list: {e}")

    def add_node(self, host: str, port: int, is_bootstrap: bool = False) -> None:
        """Add a new node to the list"""
        node_id = f"{host}:{port}"
        self.nodes[node_id] = {
            "host": host,
            "port": port,
            "is_bootstrap": is_bootstrap,
            "last_seen": time.time(),
            "status": "active"
        }
        self.save_nodes()
        self.update_queue.put(("add", node_id, self.nodes[node_id]))

    def update_node_status(self, host: str, port: int, status: str) -> None:
        """Update a node's status"""
        node_id = f"{host}:{port}"
        if node_id in self.nodes:
            self.nodes[node_id]["status"] = status
            self.nodes[node_id]["last_seen"] = time.time()
            self.save_nodes()
            self.update_queue.put(("update", node_id, self.nodes[node_id]))

    def remove_node(self, host: str, port: int) -> None:
        """Remove a node from the list"""
        node_id = f"{host}:{port}"
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.save_nodes()
            self.update_queue.put(("remove", node_id))

    def get_bootstrap_nodes(self) -> List[Tuple[str, int]]:
        """Get all bootstrap nodes"""
        bootstrap_nodes = []
        for node_id, node in self.nodes.items():
            if node["is_bootstrap"] and node["status"] == "active":
                bootstrap_nodes.append((node["host"], node["port"]))
        return bootstrap_nodes

    def get_available_nodes(self) -> List[Tuple[str, int]]:
        """Get all available nodes"""
        available_nodes = []
        for node_id, node in self.nodes.items():
            if node["status"] == "active":
                available_nodes.append((node["host"], node["port"]))
        return available_nodes

    def cleanup_inactive_nodes(self, max_age: int = 3600) -> None:
        """Remove nodes that haven't been seen for a while"""
        current_time = time.time()
        nodes_to_remove = []
        
        for node_id, node in self.nodes.items():
            if current_time - node["last_seen"] > max_age:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            del self.nodes[node_id]
            self.update_queue.put(("remove", node_id))
        
        if nodes_to_remove:
            self.save_nodes()

    def check_node(self, host: str, port: int) -> bool:
        """Check if a node is available"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except:
            return False

    def scan_network(self, start_port: int = 5001, end_port: int = 5010) -> None:
        """Scan the network for available nodes"""
        while not self.stop_event.is_set():
            try:
                # Scan localhost for nodes
                for port in range(start_port, end_port + 1):
                    if self.check_node("sterlingx.io", port):
                        # If node exists, update its status
                        if f"sterlingx.io:{port}" in self.nodes:
                            self.update_node_status("sterlingx.io", port, "active")
                        else:
                            # Add new node
                            self.add_node("sterlingx.io", port, is_bootstrap=(port == 5002))
                    else:
                        # Mark node as inactive if it exists
                        if f"sterlingx.io:{port}" in self.nodes:
                            self.update_node_status("sterlingx.io", port, "inactive")

                # Cleanup inactive nodes
                self.cleanup_inactive_nodes(max_age=60)  # Remove nodes inactive for 1 minute
                
                # Sleep for 5 seconds before next scan
                time.sleep(5)
                
            except Exception as e:
                logging.error(f"Error during network scan: {e}")
                time.sleep(5)

    def start_scanning(self, start_port: int = 5001, end_port: int = 5010) -> None:
        """Start the network scanning thread"""
        if not self.scan_thread or not self.scan_thread.is_alive():
            self.stop_event.clear()
            self.scan_thread = threading.Thread(
                target=self.scan_network,
                args=(start_port, end_port),
                daemon=True
            )
            self.scan_thread.start()

    def stop_scanning(self) -> None:
        """Stop the network scanning thread"""
        self.stop_event.set()
        if self.scan_thread:
            self.scan_thread.join()

def run_node_list_scanner(start_port: int = 5001, end_port: int = 5099, update_queue=None):
    """Run the node list scanner as a standalone process"""
    # Configure logging
    home_dir = os.path.expanduser("~")
    log_file = os.path.join(home_dir, ".local", "share", "sterlingx", "node_list_scanner.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )

    # Create node list manager
    node_list = NodeList()
    logging.info(f"Starting node list scanner for ports {start_port} to {end_port}")

    # Handle termination signals
    def signal_handler(signum, frame):
        logging.info("Received termination signal, shutting down...")
        node_list.stop_scanning()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start network scanning
        node_list.start_scanning(start_port, end_port)
        logging.info("Node list scanner started successfully")

        # Keep the process running and send updates to the queue
        while True:
            try:
                # Get updates from the queue
                while not update_queue.empty():
                    action, *args = update_queue.get()
                    if action == "add":
                        node_id, node_data = args
                        node_list.nodes[node_id] = node_data
                        node_list.save_nodes()
                    elif action == "update":
                        node_id, node_data = args
                        if node_id in node_list.nodes:
                            node_list.nodes[node_id].update(node_data)
                            node_list.save_nodes()
                    elif action == "remove":
                        node_id = args[0]
                        if node_id in node_list.nodes:
                            del node_list.nodes[node_id]
                            node_list.save_nodes()

                # Send current node list to main process
                update_queue.put(("node_list", node_list.nodes))
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error in node list scanner loop: {e}")
                time.sleep(1)

    except Exception as e:
        logging.error(f"Error in node list scanner: {e}")
        node_list.stop_scanning()
        sys.exit(1)

if __name__ == "__main__":
    run_node_list_scanner() 
