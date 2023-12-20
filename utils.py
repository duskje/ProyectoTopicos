import string
import random

import mmh3


def generate_random_car_plate(exclude) -> str:
    result = ''

    while not result or result in exclude:
        result += str(random.choice(string.ascii_uppercase))
        result += str(random.choice(string.ascii_uppercase))
        result += '-'
        result += str(random.choice(string.ascii_uppercase))
        result += str(random.choice(string.ascii_uppercase))
        result += '-'
        result += str(random.choice(list(range(10))))
        result += str(random.choice(list(range(10))))

    return result


def find_leading_zeros_for_hll(value: int, p: int):
    count = 0

    for bit_idx in reversed(range(64 - p)):
        if value & (1 << bit_idx):
            return count

        count += 1

    return count


def find_leading_zeros_for_pcsa(value: int, b: int, bitmap_length: int):
    count = 0

    for mask_idx in range(b, bitmap_length):
        if value & (1 << mask_idx):
            return count

        count += 1

if __name__ == '__main__':
    h = mmh3.hash64('AA-AA-AA', 1, False)[0]
    ldz = find_leading_zeros_for_hll(value=h, p=14)
    print(ldz)
