from __future__ import annotations
import numpy as np
from punytorch.tensor import Tensor


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

    def forward(self, *args):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, grad):
        raise NotImplementedError("Backward method not implemented.")


def ensure_numpy(x):
    if not isinstance(x, np.ndarray):
        raise TypeError("Expected a numpy array.")


class Add(Operation):
    def __init__(self, x: np.ndarray | Tensor, y: np.ndarray | Tensor):
        super().__init__()
        self.x = x.data if isinstance(x, Tensor) else x
        self.y = y.data if isinstance(y, Tensor) else y

    def forward(self):
        return self.x + self.y

    def backward(self, grad: np.ndarray):
        # grad is assumed to be a NumPy array
        # The gradient of the sum is distributed equally to both operands
        # No need to change the shape of grad since addition is element-wise
        return grad, grad


class Sub(Operation):
    def __init__(self, x: np.ndarray | Tensor, y: np.ndarray | Tensor):
        super().__init__()
        self.x = x.data if isinstance(x, Tensor) else x
        self.y = y.data if isinstance(y, Tensor) else y

    def forward(self):
        return self.x - self.y

    def backward(self, grad: np.ndarray):
        # The gradient with respect to the first operand is 1
        # The gradient with respect to the second operand is -1
        return grad, -grad


class Mul(Operation):
    def __init__(self, x: np.ndarray | Tensor, y: np.ndarray | Tensor):
        super().__init__()
        self.x = x.data if isinstance(x, Tensor) else x
        self.y = y.data if isinstance(y, Tensor) else y

    def forward(self):
        return np.multiply(self.x, self.y)

    def backward(self, grad):
        # The gradient with respect to x is y, and vice versa
        return grad * self.y, grad * self.x


class TrueDiv(Operation):
    def __init__(self, x: np.ndarray | Tensor, y: np.ndarray | Tensor):
        super().__init__(x, y)  # Pass tensors to the superclass
        # Ensure x and y are stored as numpy arrays for computation
        self.x = x.data if isinstance(x, Tensor) else x
        self.y = y.data if isinstance(y, Tensor) else y

    def forward(self):
        # Perform the division operation
        return self.x / self.y

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

    def __init__(self, x: np.ndarray | Tensor, y: np.ndarray | Tensor):
        super().__init__()
        self.x = x.data if isinstance(x, Tensor) else x
        self.y = y.data if isinstance(y, Tensor) else y

    def forward(self):
        return np.mod(self.x, self.y)

    def backward(self, grad):
        # The gradient of x % y with respect to x is 1, and with respect to y is 0
        # Check if all elements in `y.data` are integers and raise a ValueError if they're not
        if not np.all(self.y.astype(int) == self.y):
            raise ValueError("The derivative with respect to `y` is undefined for non-integer values.")
        return grad, np.zeros_like(self.y)


class Pow(Operation):
    def forward(self, x, y):
        self.x, self.y = ensure_numpy(x), ensure_numpy(y)
        return np.power(self.x, self.y)

    def backward(self, grad):
        # The gradient with respect to x is y * x^(y - 1)
        # The gradient with respect to y is x^y * log(x)
        grad_x = grad * self.y * np.power(self.x, self.y - 1)
        grad_y = grad * np.power(self.x, self.y) * np.log(self.x)
        return grad_x, grad_y


class MatMul(Operation):
    def forward(self, x, y):
        self.x, self.y = ensure_numpy(x), ensure_numpy(y)
        return np.matmul(self.x, self.y)

    def backward(self, grad):
        # If Z = X @ Y, then d(Z)/dX = grad @ Y^T and d(Z)/dY = X^T @ grad
        return np.dot(grad, self.y.T), np.dot(self.x.T, grad)


class Tanh(Operation):
    def forward(self, x):
        self.x = ensure_numpy(x)
        return np.tanh(self.x)

    def backward(self, grad):
        tanh_x = np.tanh(self.x)
        return (1 - np.square(tanh_x)) * grad
