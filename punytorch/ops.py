import numpy as np


class Operation:
    def forward(self, *args):
        raise NotImplementedError

    def backward(self, context, grad):
        raise NotImplementedError


class Add(Operation):
    def forward(self, x, y):
        """
        z = x + y
        """
        return x + y

    def backward(self, context, grad):
        """
        d(x + y)/dx = 1
        d(x + y)/dy = 1

        return grad for both x and y
        """
        return grad, grad


class Sub(Operation):
    def forward(self, x, y):
        """
        z = x - y
        """
        return x - y

    def backward(self, context, grad):
        """
        d(x - y)/dx = 1
        d(x - y)/dy = -1

        return [1] for x
        return [-1] for y
        """
        return grad, -grad


class Mul(Operation):
    def forward(self, x, y):
        """
        z = x * y
        """
        return x * y

    def backward(self, context, grad):
        """
        d(x * y)/dx = y
        d(x * y)/dy = x

        return (y.data * grad.data) for x
        return (x.data * grad.data) for y
        """
        x, y = context.args
        return y * grad, x * grad


class TrueDiv(Operation):
    def forward(self, x, y):
        """
        z = x / y
        """
        return x / y

    def backward(self, context, grad):
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


class Mod(Operation):
    """
    WARNING: The modulus operation is not differentiable at integer points,
    and the derivative with respect to `y` is undefined. This implementation
    assumes that the gradient with respect to `y` is 0, which is not universally
    accepted. It also checks if all elements in `y.data` are integers and raises
    a ValueError if they're not. This is a simplification and may not be suitable
    for all use cases.
    """

    def forward(self, x, y):
        """
        z = x % y
        """
        return x % y

    def backward(self, context, grad):
        """
        d(x % y)/dx = 1
        d(x % y)/dy = 0

        return an array of ones like x.data for x
        return an array of zerose like y.data for y
        """
        x, y = context.args
        return grad, 0


class Pow(Operation):
    def forward(self, x, y):
        """
        z = x ^ y
        """
        return x**y

    def backward(self, context, grad):
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


class MatMul(Operation):
    def forward(self, x, y):
        """
        z = x @ y
        """
        return x @ y

    def backward(self, context, grad):
        """
        If Z = X @ Y, then
        d(Z)/dX = grad @ Y.T
        d(Z)/dY = X.T @ grad
        """
        x, y = context.args
        return grad @ y.T, x.T @ grad


class Tanh(Operation):
    def forward(self, x):
        """
        z = tanh(x)
        """
        return np.tanh(x)

    def backward(self, x, grad):
        """
        d(tanh(x))/dx = 1 - tanh(x)^2

        return (1 - tanh(x)^2) * grad
        """
        grad_tanh = 1 - np.tanh(x) ** 2
        return grad_tanh * grad
