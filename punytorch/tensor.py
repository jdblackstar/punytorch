from __future__ import annotations

import numpy as np

from punytorch.activations import ReLU, Sigmoid, Softmax
from punytorch.ops import (
    Add,
    Function,
    MatMul,
    Max,
    Mean,
    Mod,
    Mul,
    Pow,
    Reshape,
    Sub,
    Sum,
    Tanh,
    TrueDiv,
)


class Tensor:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        self.requires_grad = requires_grad
        # if requires_grad is True, then we need to initialize the gradient to zeros
        # and make sure that they're floats, since backprop uses floats
        self.grad = (
            np.zeros_like(self.data, dtype=np.float64) if requires_grad else None
        )
        self.dtype = self.data.dtype
        self.context = None

    @property
    def shape(self):
        return self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # Create a new tensor from the indexed data
        result = Tensor(self.data[index], requires_grad=self.requires_grad)

        # If this tensor requires gradients, we need to set up the backward connection
        if self.requires_grad:
            # We need to create a custom indexing operation that can propagate gradients
            from punytorch.ops import Function

            class GetItem:
                @staticmethod
                def forward(x, index):
                    return x.data[index]

                @staticmethod
                def backward(context, grad):
                    x, index = context.args
                    # Create a gradient tensor of the same shape as the original
                    grad_input = np.zeros_like(x.data, dtype=np.float64)
                    # Place the gradient at the indexed location
                    grad_input[index] = grad.data if hasattr(grad, "data") else grad
                    return Tensor(grad_input), None

            result.context = Function(GetItem, self, index)

        return result

    @property
    def T(self):
        from punytorch.ops import Transpose, Function

        result = Tensor(Transpose.forward(self), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.context = Function(Transpose, self)
        return result

    def transpose(self, dim0, dim1):
        """
        Returns a tensor with dimensions dim0 and dim1 swapped.

        Args:
            dim0 (int): The first dimension to be swapped.
            dim1 (int): The second dimension to be swapped.

        Returns:
            Tensor: A tensor with dimensions dim0 and dim1 swapped.
        """
        axes = list(range(self.data.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        data = self.data.transpose(axes)
        return Tensor(data, requires_grad=self.requires_grad)

    def tolist(self):
        return self.data.tolist()

    def item(self):
        assert np.prod(self.data.shape) == 1, (
            "Only one element tensors can be converted to Python scalars"
        )
        return self.data.item()

    @staticmethod
    def data_to_numpy(data):
        if isinstance(data, (int, float)):
            return np.array([data])
        if isinstance(data, (list, tuple)):
            return np.array(data)
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, Tensor):
            return data.data.copy()
        raise ValueError(f"Invalid value passed to tensor. Type: {type(data)}")

    def clone(self):
        """
        Creates a copy of the tensor that doesn't share memory with the original tensor.

        Returns:
            Tensor: A copy of the tensor.
        """
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    @staticmethod
    def stack(tensors, axis=0):
        arrays = [t.data for t in tensors]
        stacked_array = np.stack(arrays, axis)
        return Tensor(stacked_array)

    def detach(self):
        """
        Creates a new Tensor that shares the same data but requires no gradient computation.

        Returns:
            Tensor: A new Tensor with the same data but requires_grad=False.
        """
        return Tensor(self.data, requires_grad=False)

    def backward(self, grad=None):
        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        stack = [(self, grad)]
        while stack:
            tensor, grad = stack.pop()
            if tensor.context is not None:
                grads = tensor.context.op.backward(tensor.context, grad)
                for arg, grad_arg in zip(tensor.context.args, grads):
                    if (
                        isinstance(arg, Tensor)
                        and arg.requires_grad
                        and grad_arg is not None
                    ):
                        if arg.grad is None:
                            arg.grad = np.zeros_like(arg.data)
                        arg.grad += grad_arg.data  # Ensure grad_arg is a numpy array
                        stack.append((arg, grad_arg))

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
        This context manager can temporarily disable gradient computation.
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

    def reshape(self, *shape):
        """
        Returns a tensor with the specified shape.

        Args:
            *shape: Target shape dimensions

        Returns:
            Tensor: Reshaped tensor with proper gradient tracking
        """
        if isinstance(shape[0], tuple):
            shape = shape[0]
        curr = self.prod(self.shape)
        target = self.prod((s for s in shape if s != -1))
        shape = tuple(curr // target if s == -1 else s for s in shape)

        result = Tensor(Reshape.forward(self, shape), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.context = Function(Reshape, self, shape)
        return result

    """
    REDUCTION OPS
    """

    def sum(self, axis=None, keepdims=False):
        """
        Returns the sum of tensor elements along specified axis.

        Args:
            axis: Axis or axes along which to sum. If None, sum all elements.
            keepdims: Whether to keep reduced dimensions.

        Returns:
            Tensor: Sum result with proper gradient tracking.
        """
        result = Tensor(
            Sum.forward(self, axis, keepdims), requires_grad=self.requires_grad
        )
        if self.requires_grad:
            result.context = Function(Sum, self, axis, keepdims)
        return result

    def mean(self, axis=None, keepdims=False):
        """
        Returns the mean of tensor elements along specified axis.

        Args:
            axis: Axis or axes along which to compute mean. If None, mean of all elements.
            keepdims: Whether to keep reduced dimensions.

        Returns:
            Tensor: Mean result with proper gradient tracking.
        """
        result = Tensor(
            Mean.forward(self, axis, keepdims), requires_grad=self.requires_grad
        )
        if self.requires_grad:
            result.context = Function(Mean, self, axis, keepdims)
        return result

    def max(self, axis=None, keepdims=False):
        """
        Returns the maximum of tensor elements along specified axis.

        Args:
            axis: Axis or axes along which to find max. If None, max of all elements.
            keepdims: Whether to keep reduced dimensions.

        Returns:
            Tensor: Max result with proper gradient tracking.
        """
        result = Tensor(
            Max.forward(self, axis, keepdims), requires_grad=self.requires_grad
        )
        if self.requires_grad:
            result.context = Function(Max, self, axis, keepdims)
        return result

    """
    BINARY OPS
    """

    def __add__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        result = Tensor(Add.forward(self, other))
        if self.requires_grad or other.requires_grad:
            result.context = Function(Add, self, other)
            result.requires_grad = True
        return result

    def __sub__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        result = Tensor(Sub.forward(self, other))
        if self.requires_grad or other.requires_grad:
            result.context = Function(Sub, self, other)
            result.requires_grad = True
        return result

    def __mul__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        result = Tensor(Mul.forward(self, other))
        if self.requires_grad or other.requires_grad:
            result.context = Function(Mul, self, other)
            result.requires_grad = True
        return result

    def __truediv__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        result = Tensor(TrueDiv.forward(self, other))
        if self.requires_grad or other.requires_grad:
            result.context = Function(TrueDiv, self, other)
            result.requires_grad = True
        return result

    def __mod__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        result = Tensor(Mod.forward(self, other))
        if self.requires_grad or other.requires_grad:
            result.context = Function(Mod, self, other)
            result.requires_grad = True
        return result

    def __pow__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        result = Tensor(Pow.forward(self, other))
        if self.requires_grad or other.requires_grad:
            result.context = Function(Pow, self, other)
            result.requires_grad = True
        return result

    def __matmul__(self, other):
        other = Tensor.ensure_tensor(other)
        result = Tensor(MatMul.forward(self, other))
        if self.requires_grad or other.requires_grad:
            result.context = Function(MatMul, self, other)
            result.requires_grad = True
        return result

    # Right-hand operators for scalar operations
    def __radd__(self, other) -> "Tensor":
        return self.__add__(other)

    def __rmul__(self, other) -> "Tensor":
        return self.__mul__(other)

    def __rsub__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        return other.__sub__(self)

    def __rtruediv__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        return other.__truediv__(self)

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
        result = Tensor(Tanh.forward(self.data), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.context = Function(Tanh, self)
        return result

    # TODO: implement new argmax function

    """
    ACTIVATIONS
    """

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=float)

    def relu(self):
        result = Tensor(ReLU.forward(self.data), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.context = Function(ReLU, self)
        return result

    def sigmoid(self):
        result = Tensor(Sigmoid.forward(self.data), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.context = Function(Sigmoid, self)
        return result

    def softmax(self):
        result = Tensor(Softmax.forward(self.data), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.context = Function(Softmax, self)
        return result

    def cross_entropy(self, targets):
        """
        Computes cross entropy loss between logits and targets.

        Args:
            targets (Tensor): One-hot encoded target labels

        Returns:
            Tensor: Scalar loss value with gradient computation enabled
        """
        from punytorch.losses import CrossEntropyLoss
        from punytorch.ops import Function

        targets = Tensor.ensure_tensor(targets)
        result = Tensor(CrossEntropyLoss.forward(self, targets), requires_grad=True)
        if self.requires_grad or targets.requires_grad:
            result.context = Function(CrossEntropyLoss, self, targets)
        return result

    """
    TYPING STUFF
    """

    def float(self):
        self.data = self.data.astype(np.float32)
        return self

    """
    CUSTOM METHODS FOR GENERATE FUNCTION
    """

    @staticmethod
    def zeros(shape):
        return Tensor(np.zeros(shape))

    @staticmethod
    def multinomial(input, num_samples):
        return Tensor(
            np.random.choice(range(input.shape[1]), size=num_samples, p=input.data[0])
        )

    @staticmethod
    def cat(tensors, dim=0):
        arrays = [t.data for t in tensors]
        return Tensor(np.concatenate(arrays, axis=dim))

    def long(self):
        """
        Converts the tensor to a 64-bit integer tensor.
        """
        if self.dtype is not np.int64:
            return Tensor(self.data.astype(np.int64), requires_grad=self.requires_grad)
        return self

    def to(self, device):
        """
        Moves the tensor to the specified device. Currently, only 'cpu' is supported.

        Args:
            device (str): The device to move the tensor to ('cpu' supported).

        Returns:
            Tensor: A new tensor moved to the specified device.
        """
        if device == "cpu":
            return self  # Since NumPy arrays are already on the CPU, just return self.
        else:
            raise NotImplementedError(
                "Only 'cpu' device is supported for the Tensor class."
            )
