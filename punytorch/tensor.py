from __future__ import annotations

import numpy as np

from punytorch.activations import ReLU, Sigmoid, Softmax
from punytorch.ops import Add, Function, MatMul, Mod, Mul, Pow, Sub, Tanh, TrueDiv
from punytorch.mlops import Reshape


class Tensor:
    def __init__(self, data, requires_grad=False) -> None:
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.context = None
        self.grad = np.zeros_like(self.data, dtype=float)
        self.requires_grad = requires_grad

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

    def t(self):
        return Tensor(np.transpose(self.data))

    def item(self):
        assert (
            np.prod(self.data.shape) == 1
        ), "Only one element tensors can be converted to Python scalars"
        return self.data.item()

    @staticmethod
    def data_to_numpy(data):
        if isinstance(data, (int, float)):
            return np.arary([data])
        if isinstance(data, (list, tuple)):
            return np.array(data)
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, Tensor):
            return data.data.copy()
        raise ValueError(f"Invalid value passed to tensor. Type: {type(data)}")

    def backward(self, grad=None):
        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        if self.context is not None:
            grads = self.context.op.backward(self.context, grad.data)
            for arg, grad in zip(self.context.args, grads):
                if isinstance(arg, Tensor):
                    grad = Tensor.ensure_tensor(grad)
                    arg.grad += grad
                    arg.backward(grad)
        else:
            self.grad = grad

    @staticmethod
    def ensure_tensor(data):
        if isinstance(data, Tensor):
            return data
        return Tensor(data)

    """
    ML OPS
    """

    def no_grad():
        """
        Context manager to temporarily disable gradient computation.
        """

        class NoGradContext:
            def __call__(self, func):
                def wrapper(*args, **kwargs):
                    with NoGradContext():
                        return func(*args, **kwargs)
                return wrapper

            def __enter__(self):
                Tensor._compute_grad = False

            def __exit__(self, exc_type, exc_value, traceback):
                Tensor._compute_grad = True
        return NoGradContext()
    
    def prod(self, iterable):
        result = 1
        for x in iterable:
            result *= x
        return result
    
    def reshape(self, *shape) -> Tensor:
        if isinstance(shape[0], tuple):
            shape = shape[0]
        curr = self.prod(self.shape)
        target = self.prod((s for s in shape if s != -1))
        shape = tuple(curr // target if s == -1 else s for s in shape)
        return Reshape.apply(self, shape)

    """
    BINARY OPS
    """

    def __add__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        result = Tensor(Add.forward(self.data, other.data))
        if self.requires_grad or other.requires_grad:
            result.context = Function(Add, self, other)
            result.requires_grad = True
        return result

    def __sub__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        result = Tensor(Sub.forward(self.data, other.data))
        if self.requires_grad or other.requires_grad:
            result.context = Function(Sub, self, other)
            result.requires_grad = True
        return result

    def __mul__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        result = Tensor(Mul.forward(self.data, other.data))
        if self.requires_grad or other.requires_grad:
            result.context = Function(Mul, self, other)
            result.requires_grad = True
        return result

    def __truediv__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        result = Tensor(TrueDiv.forward(self.data, other.data))
        if self.requires_grad or other.requires_grad:
            result.context = Function(TrueDiv, self, other)
            result.requires_grad = True
        return result

    def __mod__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        result = Tensor(Mod.forward(self.data, other.data))
        if self.requires_grad or other.requires_grad:
            result.context = Function(Mod, self, other)
            result.requires_grad = True
        return result

    def __pow__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        result = Tensor(Pow.forward(self.data, other.data))
        if self.requires_grad or other.requires_grad:
            result.context = Function(Pow, self, other)
            result.requires_grad = True
        return result

    def __matmul__(self, other):
        other = Tensor.ensure_tensor(other)
        result = Tensor(MatMul.forward(self.data, other.data))
        if self.requires_grad or other.requires_grad:
            result.context = Function(MatMul, self, other)
            result.requires_grad = True
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

    def tanh(self):
        result = Tensor(Tanh.forward(self.data))
        result.context = Function(Tanh, self)
        return result

    # TODO: implement new argmax function

    """
    ACTIVATIONS
    """

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=float)

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
    TYPING STUFF
    """

    def float(self):
        self.data = self.data.astype(np.float32)
        return self
