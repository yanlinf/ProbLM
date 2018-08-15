"""
The file contains strategy for cardinality estimation. The strategy we use is HyperLogLog.
Programmer: Hugo Zhang
Date: 2018/8/14
"""
import hashlib
import numpy as np


class Simple(object):
    def __init__(self, dataset=set()):
        self.dataset = set(dataset)

    def update(self, data):
        self.dataset.update(set(data))

    def estimate(self):
        return len(self.dataset)

    def merge(self, another):
        self.dataset.update(set(another.dataset))

    def clear(self):
        self.dataset = set()

    def isempty(self):
        return self.dataset == set()

    def __len__(self):
        return self.estimate()


class HyperLogLog(object):
    """
    Contains two main functions: update, estimate.
    Other functions: merge, clear, isempty.
    """
    hash_bit_size = 32

    def __init__(self, b=10, hash_func=hashlib.sha1):
        """
        :param b: log2(memory size); m = 1 << b is the memory size.
        :param hash_func: the hash function we use
        """
        assert 4 <= b <= 16, 'Memory size not appropriate. Parameter b should be in [4, 16].'
        self.b = b
        self.m = 1 << b
        self.hash_func = hash_func
        self.registers = np.zeros(self.m, dtype=int)
        self.alpha = 0.7213 / (1.0 + 1.079 / (1 << self.b))

    def update(self, data):
        """
        :param data: the data added to hyperloglog structure
        """
        leftmost = lambda bits: self.hash_bit_size - self.b - bits.bit_length() + 1
        data = int(self.hash_func(str(hash(data))).hexdigest, 16) & ((1 << self.hash_bit_size) - 1)
        index = data & (self.m - 1)
        new_result = leftmost(data >> self.b)
        assert new_result > 0, 'Hash value overflow.'
        self.registers[index] = max(self.registers[index], new_result)

    def estimate(self):
        """
        :return: the estimated cardinality
        """
        result = self.alpha * float(self.m ** 2) / np.sum(2.0 ** (-self.registers))
        if result <= 2.5 * self.m:
            num_of_zeros = self.m - np.count_nonzero(self.registers)
            if num_of_zeros != 0:
                result = self.m * np.log(self.m / float(num_of_zeros))
        elif result > (1.0 / 30.0) * (1 << 32):
            result = - (1 << 32) * np.log(1.0 - float(result) / (1 << 32))
        return result

    def merge(self, another):
        """
        :param another: to merge another hyperloglog structure
        """
        assert self.b == another.b and self.m == another.m, 'Hyperloglog size not the same.'
        self.registers = np.maximum(self.registers, another.registers)

    def clear(self):
        """
        :return: clear the registers
        """
        self.registers = np.zeros(self.m, dtype=int)

    def isempty(self):
        """
        :return: to check if the hyperloglog registers are empty
        """
        return not np.any(self.registers)

    def __len__(self):
        return self.estimate()

