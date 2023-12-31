import string

import mmh3

from utils import find_leading_zeros_for_hll, find_leading_zeros_for_pcsa


def all_possible_plates(starting: str = 'A', ending: str = 'Z'):
    starting_index = string.ascii_uppercase.find(starting.upper())
    ending_index = string.ascii_uppercase.find(ending.upper())

    for first in string.ascii_uppercase[starting_index: ending_index + 1]:
        for second in string.ascii_uppercase:
            for third in string.ascii_uppercase:
                for fourth in string.ascii_uppercase:
                    for fifth in range(10):
                        for sixth in range(10):
                            yield f'{first}{second}-{third}{fourth}-{fifth}{sixth}'


def find_plate_with_target_leading_zeros_for_hll(p: int = 14, target: int = 50):
    for plate in all_possible_plates():
        h = mmh3.hash64(plate, 1, False)[0]
        ldz = find_leading_zeros_for_hll(value=h, p=p)

        if ldz == target:
            return plate

    raise ValueError('Couldn\'t find target')


def leading_zeros_distribution_for_plates_hll(p: int = 14, starting: str = 'A', ending: str = 'Z'):
    count = {}

    for plate in all_possible_plates(starting=starting, ending=ending):
        h = mmh3.hash64(plate, 1, False)[0]
        ldz = find_leading_zeros_for_hll(value=h, p=p)

        if count.get(ldz, None) is None:
            count[ldz] = 1
        else:
            count[ldz] += 1

    return count


def find_plate_with_target_leading_zeros_for_pcsa(target, b=9, bitmap_length=32):
    for plate in all_possible_plates():
        h = mmh3.hash64(plate, 1, False)[0]
        ldz = find_leading_zeros_for_pcsa(value=h, b=b, bitmap_length=bitmap_length)

        if ldz == target:
            return plate

    raise ValueError('Couldn\'t find target')


def find_plates_until_n_leading_zeros_pcsa(n=10, b=9):
    result = {}

    for i in range(n + 1):
        try:
            plate = find_plate_with_target_leading_zeros_for_pcsa(i, b=b)
            result[i] = plate

            print(f'Found plate with {i} leading zeros: ', plate)
        except ValueError:
            print(f'Couldn\'t find a car plate with {i} leading zeros.')

    return result


def find_plates_until_n_leading_zeros_hll(n=32, p=14):
    result = {}

    for i in range(n + 1):
        try:
            plate = find_plate_with_target_leading_zeros_for_hll(p, i)
            result[i] = plate

            print(f'Found plate with {i} leading zeros: ', plate)
        except ValueError:
            print(f'Couldn\'t find a car plate with {i} leading zeros.')

    return result

    # Output
    """
Found plate with 0 leading zeros:  AA-AA-00
Found plate with 1 leading zeros:  AA-AA-07
Found plate with 2 leading zeros:  AA-AA-11
Found plate with 3 leading zeros:  AA-AA-02
Found plate with 4 leading zeros:  AA-AA-62
Found plate with 5 leading zeros:  AA-AA-05
Found plate with 6 leading zeros:  AA-AC-91
Found plate with 7 leading zeros:  AA-AB-63
Found plate with 8 leading zeros:  AA-AB-37
Found plate with 9 leading zeros:  AA-AF-88
Found plate with 10 leading zeros:  AA-AR-82
Found plate with 11 leading zeros:  AA-BM-40
Found plate with 12 leading zeros:  AA-BC-87
Found plate with 13 leading zeros:  AA-KI-72
Found plate with 14 leading zeros:  AA-NV-65
Found plate with 15 leading zeros:  AB-KN-67
Found plate with 16 leading zeros:  AA-QL-41
Found plate with 17 leading zeros:  AD-QQ-41
Found plate with 18 leading zeros:  AU-TR-68
Found plate with 19 leading zeros:  AN-OZ-20
Found plate with 20 leading zeros:  BC-HM-68
Found plate with 21 leading zeros:  DA-ZA-78
Found plate with 22 leading zeros:  HD-NT-25
Found plate with 23 leading zeros:  BL-VY-63
Found plate with 24 leading zeros:  ZE-TJ-66
Couldn't find a car plate with 25 leading zeros.
Found plate with 26 leading zeros:  AO-LP-84
Found plate with 27 leading zeros:  AM-VU-39
Couldn't find a car plate with 28 leading zeros.
Couldn't find a car plate with 29 leading zeros.
Couldn't find a car plate with 30 leading zeros.
Couldn't find a car plate with 31 leading zeros.
Couldn't find a car plate with 32 leading zeros.
    """


if __name__ == '__main__':
    # Approximately, since 2008 to date
    # print(leading_zeros_distribution_for_plates_hll(starting='b', ending='s'))
    print(find_plates_until_n_leading_zeros_pcsa(n=8))

