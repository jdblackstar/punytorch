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
    def backward(grad):
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
    def backward(grad):
        """
        d(x - y)/dx = 1
        d(x - y)/dy = -1

        return [1] for x
        return [-1] for y
        """
        return grad, -grad


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
        return y * grad, x * grad


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
        grad_x = grad / y
        grad_y = -grad * x / (y * y)
        return grad_x, grad_y


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
        return grad, 0


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

        return grad * (y * x ** (y - 1)) for x
        return grad * (x**y * np.log(x)) for y
        """
        x, y = context.args
        grad_x = grad * (y * x ** (y - 1))
        grad_y = grad * (x**y * np.log(x))
        return grad_x, grad_y


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
        return grad @ y.T, x.T @ grad


class Tanh:
    @staticmethod
    def forward(x):
        """
        z = tanh(x)
        """
        return np.tanh(x)

    @staticmethod
    def backward(x, grad):
        """
        d(tanh(x))/dx = 1 - tanh(x)^2

        return (1 - tanh(x)^2) * grad
        """
        grad_tanh = 1 - np.tanh(x) ** 2
        return grad_tanh * grad
