import math
import mmh3
import base64
import logging
import json
import random

class BloomFilter:
    def __init__(self, capacity, error_rate):
        """Initialize bloom filter"""
        try:
            self.capacity = capacity
            self.error_rate = error_rate
            self.size = self._get_size(capacity, error_rate)
            self.hash_count = self._get_hash_count(self.size, capacity)
            self.bit_array = [0] * self.size
            self.seeds = [self._generate_seed(i) for i in range(self.hash_count)]
            
            logging.info(f"Bloom filter initialized: size={self.size}, hash_count={self.hash_count}")
            
        except Exception as e:
            logging.error(f"Error initializing bloom filter: {e}")
            raise

    def _get_size(self, capacity, error_rate):
        """Calculate optimal size of bit array"""
        try:
            size = -1 * capacity * math.log(error_rate) / (math.log(2) ** 2)
            return int(size)
        except Exception as e:
            logging.error(f"Error calculating bloom filter size: {e}")
            return 256

    def _get_hash_count(self, size, capacity):
        """Calculate optimal number of hash functions"""
        try:
            count = (size / capacity) * math.log(2)
            return int(count)
        except Exception as e:
            logging.error(f"Error calculating hash count: {e}")
            return 3

    def _generate_seed(self, index):
        return random.randint(1, 1000000)

    def add(self, item):
        """Add an item to the bloom filter"""
        try:
            for i in range(self.hash_count):
                index = self._hash(item, self.seeds[i]) % self.size
                self.bit_array[index] = 1
                
        except Exception as e:
            logging.error(f"Error adding item to bloom filter: {e}")

    def _hash(self, item, seed):
        if isinstance(item, str):
            item = item.encode()
        elif isinstance(item, bytes):
            pass
        else:
            item = str(item).encode()
            
        hash_value = 0
        for byte in item:
            hash_value = (hash_value * seed + byte) & 0xFFFFFFFF
        return hash_value

    def contains(self, item):
        """Check if an item is in the bloom filter"""
        try:
            for i in range(self.hash_count):
                index = self._hash(item, self.seeds[i]) % self.size
                if self.bit_array[index] == 0:
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Error checking bloom filter: {e}")
            return False

    def to_base64(self):
        """Convert bloom filter to base64 string"""
        try:
            # Convert bit array to bytes
            byte_array = bytearray()
            for i in range(0, self.size, 8):
                byte = 0
                for j in range(8):
                    if i + j < self.size and self.bit_array[i + j]:
                        byte |= 1 << j
                byte_array.append(byte)
            
            # Convert to base64
            return base64.b64encode(byte_array).decode()
            
        except Exception as e:
            logging.error(f"Error converting bloom filter to base64: {e}")
            return ""

    @classmethod
    def from_base64(cls, base64_str, capacity, error_rate):
        """Create bloom filter from base64 string"""
        try:
            # Create new filter
            filter = cls(capacity, error_rate)
            
            # Decode base64
            byte_array = base64.b64decode(base64_str)
            
            # Convert bytes to bit array
            for i, byte in enumerate(byte_array):
                for j in range(8):
                    if i * 8 + j < filter.size:
                        filter.bit_array[i * 8 + j] = (byte >> j) & 1
            
            return filter
            
        except Exception as e:
            logging.error(f"Error creating bloom filter from base64: {e}")
            return None 