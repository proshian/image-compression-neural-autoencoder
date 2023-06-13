import json
import heapq
from collections import Counter
from typing import Dict, Any


class LooselessCompressor:
    def __init__(self, uncompressed_sequence = None, file_path: str = None):
        if file_path is not None:
            self.init_from_file(file_path)
        elif uncompressed_sequence is not None:
            self.init_from_sequence(uncompressed_sequence)
    
    def init_from_sequence(file_path: str):
        raise NotImplementedError
    
    def init_from_file(file_path: str):
        raise NotImplementedError
    
    def encode(self, symbol_sequence):
        raise NotImplementedError
    
    def encode(self, symbol_sequence):
        raise NotImplementedError
    
    def save_state_to_file(file_path):
        raise NotImplementedError



class Node:
    def __init__(self, prob, symbol = None, left = None, right = None):
        """Create node for given symbol and probability."""
        self.left = left
        self.right = right
        self.symbol = symbol
        self.prob = prob

    # Need comparator method at a minimum to work with heapq
    def __lt__(self, other):
        return self.prob < other.prob
    
    def encode(self, encoding):
        """Return bit encoding in traversal."""
        if self.left is None and self.right is None:
            yield (self.symbol, encoding)
        else:
            for v in self.left.encode(encoding + '0'):
                yield v
            for v in self.right.encode(encoding + '1'):
                yield v

class Huffman(LooselessCompressor):
    def __init__(self, uncompressed_sequence = None, file_path = None):
        super().__init__(uncompressed_sequence, file_path)
    
    def init_from_sequence(self, uncompressed_sequence):
        self.root = self.get_tree_from_sequence(uncompressed_sequence)
        self.encoding = self.get_encoding_from_tree(self.root)
    
    def init_from_file(self, file_path: str):
        self.encoding = self.get_encoding_from_file(file_path)
        self.root = self.get_tree_from_encoding(self.encoding)

    def get_encoding_from_file(self, file_path: str, key_type = int):
        with open(file_path, 'r') as f:
            encoding = json.load(f)
        encoding = {key_type(k): v for k,v in encoding.items()}
        return encoding
    
    def save_encoding_to_file(self, file_path):
        with open(file_path, 'w') as fp:
            json.dump(self.encoding, fp)

    def save_state_to_file(self, file_path):
        self.save_encoding_to_file(file_path)

    @staticmethod
    def get_encoding_from_tree(root):
        encoding = {}
        for sym, code in root.encode(''):
            encoding[sym]=code
        return encoding

    @staticmethod
    def get_tree_from_sequence(initial):
        symbol_to_num = Counter(initial)
        pq = [Node(num, symbol) for symbol, num in symbol_to_num.items()]
        heapq.heapify(pq)

        if len(pq) == 1:
            return Node(1, left=pq[0])
            # self.encoding = {pq[0].symbol: '0'}

        # Huffman Encoding Algorithm
        while len(pq) > 1:
            n1 = heapq.heappop(pq)
            n2 = heapq.heappop(pq)
            n3 = Node(prob = n1.prob + n2.prob,
                      left = n1,
                      right = n2)
            heapq.heappush(pq, n3)

        return pq[0]
    

    @staticmethod
    def get_tree_from_encoding(encoding):
        dummy_prob = None
        root = Node(dummy_prob)
        for symbol, code in encoding.items():
            node = root
            for bit in code:
                if bit == '0':
                    if node.left is None:
                        node.left = Node(dummy_prob)
                    node = node.left
                elif bit == '1':
                    if node.right is None:
                        node.right = Node(dummy_prob)
                    node = node.right
            node.symbol = symbol
        return root


    def __repr__(self):
        """Show encoding"""
        return 'huffman:' + str(self.encoding)

    def encode(self, symbol_sequence):
        """Return bit string for encoding."""
        bits = ''
        for sym in symbol_sequence:
            if not sym in self.encoding:
                sym = str(sym)
                if not sym in self.encoding:
                    raise ValueError(f"'{sym}' is not encoded character")
            bits += self.encoding[sym]
        return bits

    def decode(self, bits):
        """Decode ASCII bit string for simplicity."""
        decoded = []
        node = self.root
        for b in bits:
            if b == '0':
                node = node.left
            else:
                node = node.right

            if node.symbol:
                decoded.append(node.symbol)
                node = self.root

        return decoded