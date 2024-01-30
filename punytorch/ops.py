import numpy as np


class Function:
    def __init__(self, op, *args):
        self.op = op
        self.args = args

    def apply(self, *args):
        """
        Applies the function to the gives arguments.

        Args:
            *args: The arguments to apply the function to.

        Returns:
            The result of applying the function.
        """
        return self.forward(*args)


class Operation:
    def forward(self, *args):
        raise NotImplementedError

    def backward(self, context, grad):
        raise NotImplementedError


class Add(Operation):
    @staticmethod
    def forward(x, y):
        from punytorch.tensor import Tensor

        x_data = x.data if isinstance(x, Tensor) else np.array(x)
        y_data = y.data if isinstance(y, Tensor) else np.array(y)
        return np.add(x_data, y_data)

    @staticmethod
    def backward(context, grad):
        # grad is assumed to be a NumPy array
        # The gradient of the sum is distributed equally to both operands
        # No need to change the shape of grad since addition is element-wise
        return grad, grad


class Sub(Operation):
    @staticmethod
    def forward(x, y):
        from punytorch.tensor import Tensor

        # Ensure that x and y are NumPy arrays
        x_data = x.data if isinstance(x, Tensor) else np.array(x)
        y_data = y.data if isinstance(y, Tensor) else np.array(y)
        # Use NumPy's subtraction
        return np.subtract(x_data, y_data)

    @staticmethod
    def backward(context, grad):
        # The gradient with respect to the first operand is 1
        # The gradient with respect to the second operand is -1
        return grad, -grad


class Mul(Operation):
    @staticmethod
    def forward(x, y):
        from punytorch.tensor import Tensor

        # Ensure that x and y are NumPy arrays
        x_data = x.data if isinstance(x, Tensor) else np.array(x)
        y_data = y.data if isinstance(y, Tensor) else np.array(y)
        # Use NumPy's multiplication
        return np.multiply(x_data, y_data)

    @staticmethod
    def backward(context, grad):
        x, y = context.args
        # The gradient with respect to x is y, and vice versa
        return grad * y.data, grad * x.data


class TrueDiv(Operation):
    @staticmethod
    def forward(x, y):
        from punytorch.tensor import Tensor

        # Ensure that x and y are NumPy arrays
        x_data = x.data if isinstance(x, Tensor) else np.array(x)
        y_data = y.data if isinstance(y, Tensor) else np.array(y)
        # Use NumPy's true division
        return np.divide(x_data, y_data)

    @staticmethod
    def backward(context, grad):
        x, y = context.args
        # The gradient with respect to x is 1/y
        # The gradient with respect to y is -x/y^2
        return grad.data / y.data, -x.data * grad.data / (y.data**2)


class Mod(Operation):
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
        from punytorch.tensor import Tensor

        # Ensure that x and y are NumPy arrays
        x_data = x.data if isinstance(x, Tensor) else np.array(x)
        y_data = y.data if isinstance(y, Tensor) else np.array(y)
        # Use NumPy's mod
        return np.mod(x_data, y_data)

    @staticmethod
    def backward(context, grad):
        x, y = context.args
        # The gradient of x % y with respect to x is 1, and with respect to y is 0
        # Check if all elements in `y.data` are integers and raise a ValueError if they're not
        if not np.all(y.data.astype(int) == y.data):
            raise ValueError("The derivative with respect to `y` is undefined for non-integer values.")
        return grad, np.zeros_like(y.data)


class Pow(Operation):
    @staticmethod
    def forward(x, y):
        from punytorch.tensor import Tensor

        # Ensure that x and y are NumPy arrays
        x_data = x.data if isinstance(x, Tensor) else np.array(x)
        y_data = y.data if isinstance(y, Tensor) else np.array(y)
        # Use NumPy's power function
        return np.power(x_data, y_data)

    @staticmethod
    def backward(context, grad):
        x, y = context.args
        # The gradient with respect to x is y * x^(y - 1)
        # The gradient with respect to y is x^y * log(x)
        grad_x = grad * y.data * np.power(x.data, y.data - 1)
        grad_y = grad * np.power(x.data, y.data) * np.log(x.data)
        return grad_x, grad_y


class MatMul(Operation):
    @staticmethod
    def forward(x, y):
        from punytorch.tensor import Tensor

        # Ensure that x and y are NumPy arrays
        x_data = x.data if isinstance(x, Tensor) else np.array(x)
        y_data = y.data if isinstance(y, Tensor) else np.array(y)
        # Use NumPy's matmul
        return np.matmul(x_data, y_data)

    @staticmethod
    def backward(context, grad):
        x, y = context.args
        # If Z = X @ Y, then d(Z)/dX = grad @ Y^T and d(Z)/dY = X^T @ grad
        return grad.data @ np.transpose(y.data), np.transpose(x.data) @ grad.data


class Tanh(Operation):
    @staticmethod
    def forward(x):
        from punytorch.tensor import Tensor

        # Ensure that x is a NumPy array
        x_data = x.data if isinstance(x, Tensor) else np.array(x)
        # Use NumPy's tanh
        return np.tanh(x_data)

    @staticmethod
    def backward(context, grad):
        from punytorch.tensor import Tensor

        x = context.args[0]
        # The gradient of tanh is (1 - tanh^2(x))
        x_data = x.data if isinstance(x, Tensor) else np.array(x)
        tanh_x_data = np.tanh(x_data)
        grad_tanh = (1 - np.square(tanh_x_data)) * grad
        return grad_tanh
