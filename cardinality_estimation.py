import string
from abc import ABC, abstractmethod
from array import array
from enum import Enum, auto
from math import log
from typing import Iterable, Optional, List
from scipy.stats import bernoulli

import mmh3


class CardinalityEstimator(ABC):
    @abstractmethod
    def merge(self, other: 'CardinalityEstimator'):
        pass

    @abstractmethod
    def add(self, element: Iterable):
        pass

    @abstractmethod
    def estimate(self) -> int:
        pass


class HashFunction(Enum):
    MURMURHASH64 = auto()
    SHA512 = auto()


class HLLSketch(CardinalityEstimator):
    def __init__(self, p: int, M: Optional[array] = None, hash_function: HashFunction = HashFunction.MURMURHASH64):
        if p < 4:
            raise ValueError(f'p parameter should not be below 4 (p={p})')

        self.p = p
        self.m = 1 << p

        self.hash_function = hash_function

        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)

        if M is None:
            # Equivalent to an unsigned long long array in C
            self.M = array('Q', (0 for _ in range(2 ** p)))
        else:
            self.M = M

    @property
    def zero_count(self):
        count = 0

        for bucket in self.M:
            if bucket == 0:
                count += 1

        return count

    def estimate(self) -> int:
        """ (incomplete) """
        Z = 0

        for value in self.M:
            Z += (2 ** (-value))

        C_HLL = self.alpha * (self.m ** 2) / Z

        if C_HLL <= 2.5 * self.m:
            return int(self.m * log(self.m / self.zero_count))

        return int(C_HLL)

    def merge(self, other: 'HLLSketch'):
        """ Merge operation """
        if self.p != other.p:
            raise TypeError(f"p parameter doesn't match: self.p={self.p}, other.p={other.p}")

        M = []

        for this_bucket, other_bucket in zip(self.M, other.M):
            M.append(max(this_bucket, other_bucket))

        return HLLSketch(self.p, array('Q', M))

    def get_value(self, hash_result: int):
        mask = 0

        for bit_idx in range(64 - self.p):
            mask |= 1 << bit_idx

        return hash_result & mask

    def get_bucket(self, hash_result: int):
        mask = 0

        for bit_idx in reversed(range(64 - self.p, 64)):
            mask |= 1 << bit_idx

        return (hash_result & mask) >> (64 - self.p)

    def add(self, element: str):
        h = mmh3.hash64(element, 1, False)[0]

        bucket = self.get_bucket(h)
        value = self.get_value(h)

        leading_zeros = self.leading_zeros(value)

        if self.M[bucket] < leading_zeros + 1:
            self.M[bucket] = leading_zeros + 1

    def leading_zeros(self, value):
        count = 0

        for bit_idx in reversed(range(64 - self.p)):
            if value & (1 << bit_idx):
                return count

            count += 1

        return count


class PCSASketch(CardinalityEstimator):
    BITMAP_LENGTH = 32
    PHI = 0.77351

    def __init__(self, b: int, M: Optional[array] = None):
        self.b = b
        self.m = 2 ** self.b

        if M is None:
            # Equivalent to an unsigned long long array in C
            self.M = array('Q', (0 for _ in range(self.m)))
        else:
            self.M = M

    def get_index(self, hash_result: int):
        mask = 0

        for mask_idx in range(self.b):
            mask |= 1 << mask_idx

        return hash_result & mask

    def get_leading_zeroes(self, hash_result: int):
        count = 0

        for mask_idx in range(self.b, self.BITMAP_LENGTH):
            if hash_result & (1 << mask_idx):
                return count

            count += 1

        return count

    def add(self, element: str):
        h = mmh3.hash(element, 1, False)

        index = self.get_index(h)
        value = self.get_leading_zeroes(h)

        self.M[index] |= 1 << value

    def get_least_significant_zero(self, bitmap: int):
        count = 0

        for mask_idx in range(self.BITMAP_LENGTH):
            if bitmap & (1 << mask_idx):
                count += 1

        return count

    def estimate(self) -> int:
        average_least_significant_zeros = 0

        for bitmap in self.M:
            average_least_significant_zeros += self.get_least_significant_zero(bitmap)

        average_least_significant_zeros /= self.m

        return int((self.m / self.PHI) * (2 ** average_least_significant_zeros))

    def merge(self, other):
        M = []

        for this_bitmap, other_bitmap in zip(self.M, other.M):
            M.append(this_bitmap & other_bitmap)

        return PCSASketch(self.b, array('Q', M))


def l_func_first_derivative(n, T: List[List[int]], p: float, q: float):
    l = 0

    for i in range(len(T)):
        for j in range(len(T[0])):
            pho_ij = (2 ** (-j - 1)) / len(T)
            gamma_j = 1 - pho_ij
            gamma_j_n = gamma_j ** n

            if not T[i][j]:
                l += ((p - q) * gamma_j_n * log(gamma_j)) / (1 - p + (p - q) * gamma_j_n)
            else:
                l -= ((p - q) * gamma_j_n * log(gamma_j)) / (p - (p - q) * gamma_j_n)

    return l


def l_func_second_derivative(n, T: List[List[int]], p: float, q: float):
    l = 0

    for i in range(len(T)):
        for j in range(len(T[0])):
            pho_ij = (2 ** (-j - 1)) / len(T)
            gamma_j = 1 - pho_ij
            gamma_j_n = gamma_j ** n

            if not T[i][j]:
                l += ((1 - p) * (p - q) * gamma_j_n * (log(gamma_j) ** 2)) / ((1 - p + (p - q) * gamma_j_n) ** 2)
            else:
                l -= (p * (p - q) * gamma_j_n * (log(gamma_j) ** 2)) / ((p - (p - q) * gamma_j_n) ** 2)

    return l


class SketchFlipMerge:
    BITMAP_LENGTH = 32
    NEWTON_ITERS = 200

    def __init__(self, b: int, p: float, M: Optional[array] = None):
        self.b = b
        self.m = 2 ** self.b

        if p < 0 or p > 1:
            raise ValueError(f'p parameter must be between 0 and 1 (p={p})')

        self.p = p
        self.q = 1 - self.p

        if M is None:
            # Equivalent to an unsigned long array in C
            self.M = array('L', (0 for _ in range(self.m)))

            for bitmap in self.M:
                for value_idx in range(self.b):
                    bitmap |= bernoulli.rvs(self.q) << value_idx
        else:
            self.M = M

    def get_index(self, hash_result: int):
        mask = 0

        for mask_idx in range(self.b):
            mask |= 1 << mask_idx

        return hash_result & mask

    def get_leading_zeroes(self, hash_result: int):
        count = 0

        for mask_idx in range(self.b, self.BITMAP_LENGTH):
            if hash_result & (1 << mask_idx):
                return count

            count += 1

        return count

    def add(self, element: str):
        h = mmh3.hash(element, 1, False)

        index = self.get_index(h)
        value = self.get_leading_zeroes(h)

        self.M[index] |= bernoulli.rvs(self.p) << value

    def estimate(self):
        n = 1

        T = []

        for bitmap in self.M:
            new_row = []

            for i in range(self.b):
                new_row.append(1 if bitmap & (1 << i) else 0)

            T.append(new_row)

        for _ in range(self.NEWTON_ITERS):
            n = n - l_func_first_derivative(n, T, self.p, self.q) / l_func_second_derivative(n, T, self.p, self.q)

        return int(n)


if __name__ == '__main__':
    from random import choice, randint

    random_strs = []

    for _ in range(50000):
        random_str = ''

        for _ in range(randint(5, 15)):
            random_str += choice(string.ascii_letters)

        if random_str not in random_strs:
            random_strs.append(random_str)

    pcsa_sketch = PCSASketch(8)

    for random_str in random_strs:
        pcsa_sketch.add(random_str)

    print(pcsa_sketch.estimate())

    sketch_flip_merge = SketchFlipMerge(8, p=1)

    for random_str in random_strs:
        sketch_flip_merge.add(random_str)

    print(sketch_flip_merge.estimate())
    print(len(random_strs))

