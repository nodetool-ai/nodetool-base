import numpy as np
from .utils import BinaryOperation


class AddArray(BinaryOperation):
    """
    Performs addition on two arrays.
    math, plus, add, addition, sum, +
    """

    def operation(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.add(a, b)


class SubtractArray(BinaryOperation):
    """
    Subtracts the second array from the first.
    math, minus, difference, -
    """

    def operation(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.subtract(a, b)


class MultiplyArray(BinaryOperation):
    """
    Multiplies two arrays.
    math, product, times, *
    """

    def operation(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.multiply(a, b)


class DivideArray(BinaryOperation):
    """
    Divides the first array by the second.
    math, division, arithmetic, quotient, /
    """

    def operation(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.divide(a, b)


class ModulusArray(BinaryOperation):
    """
    Calculates the element-wise remainder of division.
    math, modulo, remainder, mod, %

    Use cases:
    - Implementing cyclic behaviors
    - Checking for even/odd numbers
    - Limiting values to a specific range
    """

    def operation(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.mod(a, b)
