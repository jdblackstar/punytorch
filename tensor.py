import numpy as np

from activations import ReLU, Sigmoid, Softmax
from ops import Add, Function, MatMul, Mod, Mul, Pow, Sub, TrueDiv


class Tensor:
    def __init__(self, data) -> None:
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.context = None
        self.grad = None

    """
    ML OPS
    """

    def backward(self, grad=None):
        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        self.grad = grad
        if self.context is not None:
            grads = self.context.op.backward(
                self.context, grad.data
            )  # pass Function object as context
            for tensor, grad in zip(self.context.args, grads):
                if tensor.grad is None:
                    tensor.grad = grad
                else:
                    tensor.grad += grad

    def relu(self):
        result = Tensor(ReLU.forward(self.data))
        result.context = Function(ReLU, self)
        return result

    def sigmoid(self):
        result = Tensor(Sigmoid.forward(self.data))
        result.context = Function(Sigmoid, self)
        return result

    def softmax(self):
        result = Tensor(Softmax.forward(self.data))
        result.context = Function(Softmax, self)
        return result

    """
    BINARY OPS
    """

    def __add__(self, other) -> "Tensor":
        result = Tensor(Add.forward(self.data, other.data))
        result.context = Function(Add, self, other)
        return result

    def __sub__(self, other) -> "Tensor":
        result = Tensor(Sub.forward(self.data, other.data))
        result.context = Function(Sub, self, other)
        return result

    def __mul__(self, other) -> "Tensor":
        result = Tensor(Mul.forward(self.data, other.data))
        result.context = Function(Mul, self, other)
        return result

    def __truediv__(self, other) -> "Tensor":
        result = Tensor(TrueDiv.forward(self.data, other.data))
        result.context = Function(TrueDiv, self, other)
        return result

    def __mod__(self, other) -> "Tensor":
        fn = Function(Mod, self, other)
        result = Tensor(Mod.forward(self.data, other.data))
        result.context = fn
        return result

    def __pow__(self, other) -> "Tensor":
        result = Tensor(Pow.forward(self.data, other.data))
        result.context = Function(Pow, self, other)
        return result

    def __matmul__(self, other):
        result = Tensor(MatMul.forward(self.data, other.data))
        result.context = Function(MatMul, self, other)
        return result

    """
    UNARY OPS
    """

    def __abs__(self) -> "Tensor":
        return Tensor(np.abs(self.data))

    def __neg__(self) -> "Tensor":
        return Tensor(-self.data)

    def __invert__(self) -> "Tensor":
        return Tensor(~self.data)

    def __repr__(self) -> str:
        return f"tensor({self.data})"
