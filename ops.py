import numpy as np


class Function:
    def __init__(self, op, *args):
        self.op = op
        self.args = args


class Add:
    @staticmethod
    def forward(x, y):
        """
        z = x + y
        """
        return x + y

    @staticmethod
    def backward(context, grad):
        """
        d(x + y)/dx = 1
        d(x + y)/dy = 1
        
        return [1] for both x and y
        """
        x, y = context.args
        return [1], [1]


class Sub:
    @staticmethod
    def forward(x, y):
        """
        z = x - y
        """
        return x - y

    @staticmethod
    def backward(context, grad):
        """
        d(x - y)/dx = 1
        d(x - y)/dy = -1
        
        return [1] for x
        return [-1] for y
        """
        x, y = context.args
        return [1], [-1]


class Mul:
    @staticmethod
    def forward(x, y):
        """
        z = x * y
        """
        return x * y

    @staticmethod
    def backward(context, grad):
        """
        d(x * y)/dx = y
        d(x * y)/dy = x
        
        return (y.data * grad.data) for x
        return (x.data * grad.data) for y
        """
        x, y = context.args
        return (y.data * grad.data), (x.data * grad.data)


class TrueDiv:
    @staticmethod
    def forward(x, y):
        """
        z = x / y
        """
        return x / y

    @staticmethod
    def backward(context, grad):
        """
        d(x / y)/dx = 1/y
        d(x / y)/dy = -x/y^2

        return (grad.data / y.data) for x
        return (-grad.data * x.data / (y.data ** 2)) for y
        """
        x, y = context.args
        return (grad.data / y.data), (-grad.data * x.data / (y.data ** 2))


class Mod:
    """
    WARNING: The modulus operation is not differentiable at integer points,
    and the derivative with respect to `y` is undefined. This implementation
    assumes that the gradient with respect to `y` is 0, which is not universally
    accepted. It also checks if all elements in `y.data` are integers and raises
    a ValueError if they're not. This is a simplification and may not be suitable
    for all use cases.
    """

    @staticmethod
    def forward(x, y):
        """
        z = x % y
        """
        return x % y

    @staticmethod
    def backward(context, grad):
        """
        d(x % y)/dx = 1
        d(x % y)/dy = 0

        return an array of ones like x.data for x
        return an array of zerose like y.data for y
        """
        x, y = context.args
        if not np.all(y.data.astype(int) == y.data):
            raise ValueError(
                "Backward operation for modulus is only defined when y is an integer."
            )
        return (np.ones_like(x.data)), (np.zeros_like(y.data))


class Pow:
    @staticmethod
    def forward(x, y):
        """
        z = x ^ y
        """
        return x ** y

    @staticmethod
    def backward(context, grad):
        """
        d(x ^ y)/dx = y * x^(y - 1)
        d(x ^ y)/dy = x^y * log(x)

        return (y.data * x.data ** (y.data - 1)) for x
        return (x.data**y.data * np.log(x.data)) for y
        """
        x, y = context.args
        return (y.data * x.data ** (y.data - 1)), (x.data**y.data * np.log(x.data))

class MatMul:
    @staticmethod
    def forward(x, y):
        """
        z = x @ y
        """
        return x @ y  # @ is the matrix multiplication operator in Python

    @staticmethod
    def backward(context, grad):
        """
        If Z = X @ Y, then
        d(Z)/dX = grad @ Y.T
        d(Z)/dY = X.T @ grad
        """
        x, y = context.args

        # Shape Check
        assert x.data.shape[1] == y.data.shape[0], "Incompatible shapes for matrix multiplication"
        assert grad.shape[0] == x.data.shape[0], "Incompatible shapes for backward matrix multiplication"
        assert grad.shape[1] == y.data.shape[1], "Incompatible shapes for backward matrix multiplication"

        # Dimension Check
        assert x.data.ndim >= 2 and y.data.ndim >= 2, "Both inputs to matmul should be at least 2-dimensional"
        assert grad.ndim >= 2, "The gradient should be at least 2-dimensional"

        # Non-emptiness Check
        assert x.data.size > 0 and y.data.size > 0 and grad.size > 0, "Inputs to matmul should not be empty"

        return grad @ y.data.T, x.data.T @ grad