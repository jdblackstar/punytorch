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
        return self.op(*args)


class Operation:
    """
    Base class for all operations in the punytorch framework.

    This class provides the basic structure for defining operations that can be applied to tensors.
    Operations are callable objects that, when called, perform a forward computation and store
    information necessary for the backward pass in the computational graph.

    Attributes:
        operands (tuple): A tuple of operands (tensors or other operations) that the operation acts upon.
        inputs (tuple): A tuple of inputs provided to the operation during the forward pass.
        outputs (any): The result of the forward computation. The type depends on the specific operation.
    """

    def __init__(self, *operands):
        self.operands = operands

    def __call__(self, *args):
        self.inputs = args
        self.outputs = self.forward(*args)
        return self.outputs

    def extract_data(self, x):
        return x.data if hasattr(x, "data") else x

    def forward(self, *args):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, grad):
        raise NotImplementedError("Backward method not implemented.")


def ensure_numpy(x):
    if not isinstance(x, np.ndarray):
        raise TypeError("Expected a numpy array.")


class Add(Operation):
    def __init__(self, x, y):
        super().__init__()
        self.x = self.extract_data(x)
        self.y = self.extract_data(y)

    def forward(self):
        return np.add(self.x, self.y)

    def backward(self, grad):
        return grad, grad


class Sub(Operation):
    def __init__(self, x, y):
        super().__init__()
        self.x = self.extract_data(x)
        self.y = self.extract_data(y)

    def forward(self):
        return np.subtract(self.x, self.y)

    def backward(self, grad):
        # The gradient with respect to the first operand is 1
        # The gradient with respect to the second operand is -1
        return grad, -grad


class Mul(Operation):
    def __init__(self, x, y):
        super().__init__()
        self.x = self.extract_data(x)
        self.y = self.extract_data(y)

    def forward(self):
        return np.multiply(self.x, self.y)

    def backward(self, grad):
        # The gradient with respect to x is y, and vice versa
        return grad * self.y, grad * self.x


class TrueDiv(Operation):
    def __init__(self, x, y):
        super().__init__()
        self.x = self.extract_data(x)
        self.y = self.extract_data(y)

    def forward(self):
        return np.true_divide(self.x, self.y)

    def backward(self, grad):
        # The gradient with respect to x is 1/y
        grad_x = grad / self.y
        # The gradient with respect to y is -x/y^2
        grad_y = -self.x * grad / (self.y**2)
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

    def __init__(self, x, y):
        super().__init__()
        self.x = self.extract_data(x)
        self.y = self.extract_data(y)

    def forward(self):
        return np.mod(self.x, self.y)

    def backward(self, grad):
        # The gradient of x % y with respect to x is 1, and with respect to y is 0
        # Check if all elements in `y.data` are integers and raise a ValueError if they're not
        if not np.all(self.y.astype(int) == self.y):
            raise ValueError("The derivative with respect to `y` is undefined for non-integer values.")
        return grad, np.zeros_like(self.y)


class Pow(Operation):
    def __init__(self, x, y):
        super().__init__()
        self.x = self.extract_data(x)
        self.y = self.extract_data(y)

    def forward(self):
        return np.power(self.x, self.y)

    def backward(self, grad):
        # The gradient with respect to x is y * x^(y - 1)
        # The gradient with respect to y is x^y * log(x)
        grad_x = grad * self.y * np.power(self.x, self.y - 1)
        grad_y = grad * np.power(self.x, self.y) * np.log(self.x)
        return grad_x, grad_y


class MatMul(Operation):
    def __init__(self, x, y):
        super().__init__()
        self.x = self.extract_data(x)
        self.y = self.extract_data(y)

    def forward(self):
        return np.matmul(self.x, self.y)

    def backward(self, grad):
        # The gradient of Z = X @ Y with respect to X is grad @ Y^T
        # The gradient of Z = X @ Y with respect to Y is X^T @ grad
        return np.dot(grad, self.y.T), np.dot(self.x.T, grad)


class Tanh(Operation):
    def __init__(self, x):
        super().__init__()
        self.x = self.extract_data(x)

    def forward(self):
        return np.tanh(self.x)

    def backward(self, grad):
        # The gradient of tanh(x) with respect to x is 1 - tanh(x)^2
        tanh_x = np.tanh(self.x)
        return (1 - np.square(tanh_x)) * grad
