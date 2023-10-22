import numpy as np

from ops import Add, Function, Mod, Mul, Pow, Sub, TrueDiv


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
            grads = self.context.op.backward(self.context, grad.data)  # pass Function object as context
            for tensor, grad in zip(self.context.args, grads):
                if tensor.grad is None:
                    tensor.grad = grad
                else:
                    tensor.grad += grad

    """
    BINARY OPS
    """

    def __add__(self, other) -> "Tensor":
        fn = Function(Add, self, other)
        result = Tensor(Add.forward(self.data, other.data))
        result.context = fn
        return result

    def __sub__(self, other) -> "Tensor":
        fn = Function(Sub, self, other)
        result = Tensor(Sub.forward(self.data, other.data))
        result.context = fn
        return result

    def __mul__(self, other) -> "Tensor":
        fn = Function(Mul, self, other)
        result = Tensor(Mul.forward(self.data, other.data))
        result.context = fn
        return result

    def __truediv__(self, other) -> "Tensor":
        fn = Function(TrueDiv, self, other)
        result = Tensor(TrueDiv.forward(self.data, other.data))
        result.context = fn
        return result

    def __mod__(self, other) -> "Tensor":
        fn = Function(Mod, self, other)
        result = Tensor(Mod.forward(self.data, other.data))
        result.context = fn
        return result

    def __pow__(self, other) -> "Tensor":
        fn = Function(Pow, self, other)
        result = Tensor(Pow.forward(self.data, other.data))
        result.context = fn
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
