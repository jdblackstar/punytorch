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

        return grad for both x and y
        """
        return grad, grad


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
        return (np.array(grad.data) / y.data), (
            -np.array(grad.data) * x.data / (y.data**2)
        )


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
        return x**y

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
        return x @ y

    @staticmethod
    def backward(context, grad):
        """
        If Z = X @ Y, then
        d(Z)/dX = grad @ Y.T
        d(Z)/dY = X.T @ grad
        """
        x, y = context.args
        return grad @ y.data.T, x.data.T @ grad


class Tanh:
    @staticmethod
    def forward(x):
        return np.tanh(x.data)

    @staticmethod
    def backward(context, grad):
        (x,) = context.args
        grad_tanh = 1 - np.tanh(x.data) ** 2
        return grad_tanh * grad
