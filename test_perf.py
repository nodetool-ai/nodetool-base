import time
import math
from functools import reduce

def test_sum_old(values):
    if not values:
        raise ValueError("Cannot sum empty list")
    if not all(isinstance(x, (int, float)) for x in values):
        raise ValueError("All values must be numbers")
    return sum(values)

def test_sum_new(values):
    if not values:
        raise ValueError("Cannot sum empty list")
    try:
        res = sum(values)
        if not isinstance(res, (int, float)):
             raise ValueError("All values must be numbers")
        return res
    except TypeError:
        raise ValueError("All values must be numbers")

def test_min_old(values):
    if not values:
        raise ValueError("Cannot find minimum of empty list")
    if not all(isinstance(x, (int, float)) for x in values):
        raise ValueError("All values must be numbers")
    return min(values)

def test_min_new(values):
    if not values:
        raise ValueError("Cannot find minimum of empty list")
    try:
        res = min(values)
        if not isinstance(res, (int, float)):
             raise ValueError("All values must be numbers")
        return res
    except TypeError:
        raise ValueError("All values must be numbers")

values = list(range(1000000))

start = time.time()
test_sum_old(values)
print("Sum Old:", time.time() - start)

start = time.time()
test_sum_new(values)
print("Sum New:", time.time() - start)

start = time.time()
test_min_old(values)
print("Min Old:", time.time() - start)

start = time.time()
test_min_new(values)
print("Min New:", time.time() - start)
