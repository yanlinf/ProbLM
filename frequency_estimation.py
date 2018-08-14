"""
The file contains strategies for frequency estimation, including Count Sketch, Count-Min Sketch.

Programmer: Hugo Zhang
Date: 2018/8/13
"""
import hashlib
import numpy as np
from collections import Counter


class Sketch(object):
    """
    Base class for two strategies.
    Contains two functions: process, query(can be used by [] operator).
    """

    def __init__(self, hash_size, hash_num):
        """
        :param hash_size: the size of hash table, 2/epsilon (or 3/squared epsilon)
        :param hash_num: the amount of hash tables, O(log(1/delta))
        """
        assert isinstance(hash_size, int) and hash_size > 0, \
            'The size of hash table should be positive integer.'
        assert isinstance(hash_num, int) and hash_num > 0, \
            'The amount of hash tables should be positive integer.'

        self.hash_size = hash_size
        self.hash_num = hash_num
        self.counters = np.zeros((hash_num, hash_size), dtype=int)

    def myhash(self, x, hash_func=hash):
        """
        :param x: element to be hashed
        :param hash_func: hash function from 2-universal family
        use 'yield' to itemize
        """
        if hash_func == hash:
            for i in range(self.hash_num):
                yield hash(str(x) + str(i)) % self.hash_size
        else:
            hashing = hash_func(str(hash(x)).encode())
            for i in range(self.hash_num):
                hashing.update(str(i).encode())
                yield int(hashing.hexdigest(), 16) % self.hash_size

    def process(self, x, c=1):
        """
        :param x: element to be counted
        :param c: addend
        """
        pass

    def query(self, x):
        """
        :param x: the element to be counted
        :return: the estimated frequency of x
        """
        pass

    def __getitem__(self, x):
        """
        [] operator implement for query
        """
        return self.query(x)

    """
    def __setitem__(self, x, c):
        return self.process(self, x, c)
    """


class Simple(Sketch):

    def __init__(self):
        self.counters = Counter()

    def __iadd__(self, other):
        self.counters += other.counters
        return self

    def process(self, x, c=1):
        self.counters[x] += c

    def query(self, x):
        return self.counters[x]


class CountSketch(Sketch):

    def __init__(self, hash_size, hash_num):
        super().__init__(hash_size, hash_num)

    def __iadd__(self, other):
        self.counters += other.counters
        return self

    def myhash2(self, x, hash_func=hash):
        if hash_func == hash:
            for i in range(self.hash_num):
                yield (hash(str(x) + str(i)) % 2) * 2 - 1
        else:
            hashing = hash_func(str(hash(x)))
            for i in range(self.hash_num):
                hashing.update(str(i))
                yield (int(hashing.hexdigest(), 16) % 2) * 2 - 1

    def process(self, x, c=1):
        assert isinstance(c, int) and c > 0, \
            'The times of occurrence should be positive integer.'
        for row, h1, h2 in zip(self.counters, self.myhash(x), self.myhash2(x)):
            row[h1] += (c * h2)

    def query(self, x):
        result = [h2 * row[h1] for row, h1,
                  h2 in zip(self.counters, self.myhash(x), self.myhash2(x))]
        return np.median(result)


class CountMinSketch(Sketch):

    def __init__(self, hash_size, hash_num):
        super().__init__(hash_size, hash_num)

    def __iadd__(self, other):
        self.counters += other.counters
        return self

    def process(self, x, c=1):
        assert isinstance(c, int) and c > 0, \
            'The times of occurrence should be positive integer.'
        for row, h in zip(self.counters, self.myhash(x)):
            row[h] += c

    def query(self, x):
        return min(row[h] for row, h in zip(self.counters, self.myhash(x)))
