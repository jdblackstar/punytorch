import numpy as np


def maximum(x, y):
    assert len(x) == len(y), "Input lists must have the same length"
    return [max(a, b) for a, b in zip(x, y)]


def sum(x):
    sum_val = 0
    for i in range(len(x)):
        sum_val += x[i]
    return sum_val


def exp(x, n=10):
    """
    Compute the exponential of x using a power series expansion.
    """
    result = 1.0
    power = 1.0
    factorial = 1.0
    for i in range(1, n):
        power *= x
        factorial *= i
        result += power / factorial
    return result


def is_one_hot(labels):
    # Check if the labels are 2D.
    if len(labels.shape) != 2:
        return False

    # Check if the labels are binary vectors with a single 1.
    for label in labels:
        if np.sum(label) != 1 or np.max(label) != 1 or np.min(label) != 0:
            return False

    return True


class CharTokenizer:
    def __init__(self, text=None, filepath=None):
        self.text = text
        if filepath:
            with open(filepath, "r", encoding="utf-8") as f:
                self.text = f.read()
        elif text is None:
            raise ValueError("Either text or filepath must be provided.")

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_int = {c: i for i, c in enumerate(self.chars)}
        self.int_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_int[c] for c in text]

    def decode(self, encoded_chars):
        return "".join([self.int_to_char[i] for i in encoded_chars])
