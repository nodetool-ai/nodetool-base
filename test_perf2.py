import time
import math
from functools import reduce

def test_prod_old(values):
    if not values:
        raise ValueError("Cannot calculate product of empty list")
    if not all(isinstance(x, (int, float)) for x in values):
        raise ValueError("All values must be numbers")
    return reduce(lambda x, y: x * y, values)

def test_prod_new(values):
    if not values:
        raise ValueError("Cannot calculate product of empty list")
    try:
        res = math.prod(values)
        if not isinstance(res, (int, float)):
             raise ValueError("All values must be numbers")
        return res
    except TypeError:
        raise ValueError("All values must be numbers")

values = [1.0001] * 1000000

start = time.time()
test_prod_old(values)
print("Prod Old:", time.time() - start)

start = time.time()
test_prod_new(values)
print("Prod New:", time.time() - start)
