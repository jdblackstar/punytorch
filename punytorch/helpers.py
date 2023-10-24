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
