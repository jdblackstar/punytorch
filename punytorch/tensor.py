from __future__ import annotations

import numpy as np

from punytorch.activations import ReLU, Sigmoid, Softmax
from punytorch.ops import (
    Abs,
    Add,
    Cat,
    Function,
    MatMul,
    Max,
    Mean,
    Mod,
    Mul,
    Pow,
    Reshape,
    Stack,
    Sub,
    Sum,
    Tanh,
    TrueDiv,
)


class Tensor:
    _compute_grad = True

    def __init__(self, data, requires_grad=False):
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        self.requires_grad = requires_grad
        # if requires_grad is True, then we need to initialize the gradient to zeros
        # and make sure that they're floats, since backprop uses floats
        self.grad = np.zeros_like(self.data, dtype=np.float64) if requires_grad else None
        self.dtype = self.data.dtype
        self.context = None

    @staticmethod
    def _should_track(*values):
        return Tensor._compute_grad and any(isinstance(value, Tensor) and value.requires_grad for value in values)

    @staticmethod
    def _match_grad_shape(grad, shape):
        if isinstance(grad, Tensor):
            grad = grad.data
        grad = np.asarray(grad, dtype=np.float64)
        if grad.shape == shape:
            return grad
        if shape == ():
            return np.asarray(grad.sum(), dtype=np.float64)
        if grad.shape == ():
            return np.broadcast_to(grad, shape).astype(np.float64)
        return grad.reshape(shape)

    @property
    def shape(self):
        return self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        index_data = index.data if isinstance(index, Tensor) else index
        # Create a new tensor from the indexed data
        track = Tensor._should_track(self)
        result = Tensor(self.data[index_data], requires_grad=track)

        # If this tensor requires gradients, we need to set up the backward connection
        if track:
            # We need to create a custom indexing operation that can propagate gradients
            class GetItem:
                @staticmethod
                def forward(x, index):
                    return x[index]

                @staticmethod
                def backward(context, grad):
                    x, index = context.args
                    # Create a gradient tensor of the same shape as the original
                    grad_input = np.zeros_like(x.data, dtype=np.float64)
                    # Place the gradient at the indexed location
                    grad_data = np.asarray(grad, dtype=np.float64)
                    try:
                        np.add.at(grad_input, index, grad_data)
                    except TypeError:
                        grad_input[index] += grad_data
                    return grad_input, None

            result.context = Function(GetItem, self, index_data)

        return result

    @property
    def T(self):
        from punytorch.ops import Transpose, Function

        track = Tensor._should_track(self)
        result = Tensor(Transpose.forward(self.data), requires_grad=track)
        if track:
            result.context = Function(Transpose, self, None)
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
        axes = tuple(axes)
        from punytorch.ops import Transpose

        track = Tensor._should_track(self)
        result = Tensor(Transpose.forward(self.data, axes), requires_grad=track)
        if track:
            result.context = Function(Transpose, self, axes)
        return result

    def tolist(self):
        return self.data.tolist()

    def item(self):
        assert np.prod(self.data.shape) == 1, "Only one element tensors can be converted to Python scalars"
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
        tensors = [Tensor.ensure_tensor(t) for t in tensors]
        track = Tensor._should_track(*tensors)
        result = Tensor(Stack.forward(*[t.data for t in tensors], axis), requires_grad=track)
        if track:
            result.context = Function(Stack, *tensors, axis)
        return result

    def detach(self):
        """
        Creates a new Tensor that shares the same data but requires no gradient computation.

        Returns:
            Tensor: A new Tensor with the same data but requires_grad=False.
        """
        return Tensor(self.data, requires_grad=False)

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data, dtype=np.float64)
        else:
            grad = Tensor._match_grad_shape(grad, self.data.shape)

        stack = [(self, grad)]
        while stack:
            tensor, grad = stack.pop()
            grad_data = Tensor._match_grad_shape(grad, tensor.data.shape)
            if tensor.requires_grad:
                if tensor.grad is None:
                    tensor.grad = np.zeros_like(tensor.data, dtype=np.float64)
                tensor.grad += grad_data
            if tensor.context is not None:
                grads = tensor.context.op.backward(tensor.context, grad_data)
                for arg, grad_arg in zip(tensor.context.args, grads):
                    if isinstance(arg, Tensor) and arg.requires_grad and grad_arg is not None:
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
                self.prev = Tensor._compute_grad
                Tensor._compute_grad = False

            def __exit__(self, exc_type, exc_value, traceback):
                Tensor._compute_grad = self.prev

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

        track = Tensor._should_track(self)
        result = Tensor(Reshape.forward(self.data, shape), requires_grad=track)
        if track:
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
        track = Tensor._should_track(self)
        result = Tensor(Sum.forward(self.data, axis, keepdims), requires_grad=track)
        if track:
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
        track = Tensor._should_track(self)
        result = Tensor(Mean.forward(self.data, axis, keepdims), requires_grad=track)
        if track:
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
        track = Tensor._should_track(self)
        result = Tensor(Max.forward(self.data, axis, keepdims), requires_grad=track)
        if track:
            result.context = Function(Max, self, axis, keepdims)
        return result

    """
    BINARY OPS
    """

    def __add__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        track = Tensor._should_track(self, other)
        result = Tensor(Add.forward(self.data, other.data), requires_grad=track)
        if track:
            result.context = Function(Add, self, other)
        return result

    def __sub__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        track = Tensor._should_track(self, other)
        result = Tensor(Sub.forward(self.data, other.data), requires_grad=track)
        if track:
            result.context = Function(Sub, self, other)
        return result

    def __mul__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        track = Tensor._should_track(self, other)
        result = Tensor(Mul.forward(self.data, other.data), requires_grad=track)
        if track:
            result.context = Function(Mul, self, other)
        return result

    def __truediv__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        track = Tensor._should_track(self, other)
        result = Tensor(TrueDiv.forward(self.data, other.data), requires_grad=track)
        if track:
            result.context = Function(TrueDiv, self, other)
        return result

    def __mod__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        track = Tensor._should_track(self, other)
        result = Tensor(Mod.forward(self.data, other.data), requires_grad=track)
        if track:
            result.context = Function(Mod, self, other)
        return result

    def __pow__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        track = Tensor._should_track(self, other)
        result = Tensor(Pow.forward(self.data, other.data), requires_grad=track)
        if track:
            result.context = Function(Pow, self, other)
        return result

    def __matmul__(self, other):
        other = Tensor.ensure_tensor(other)
        track = Tensor._should_track(self, other)
        result = Tensor(MatMul.forward(self.data, other.data), requires_grad=track)
        if track:
            result.context = Function(MatMul, self, other)
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

    def __rpow__(self, other) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        return other.__pow__(self)

    """
    UNARY OPS
    """

    def __abs__(self) -> "Tensor":
        track = Tensor._should_track(self)
        result = Tensor(Abs.forward(self.data), requires_grad=track)
        if track:
            result.context = Function(Abs, self)
        return result

    def __neg__(self) -> "Tensor":
        return self * -1

    def __invert__(self) -> "Tensor":
        return Tensor(~self.data)

    def __repr__(self) -> str:
        return f"tensor({self.data})"

    def tanh(self):
        track = Tensor._should_track(self)
        result = Tensor(Tanh.forward(self.data), requires_grad=track)
        if track:
            result.context = Function(Tanh, self)
        return result

    # TODO: implement new argmax function

    """
    ACTIVATIONS
    """

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=float)

    def relu(self):
        track = Tensor._should_track(self)
        result = Tensor(ReLU.forward(self.data), requires_grad=track)
        if track:
            result.context = Function(ReLU, self)
        return result

    def sigmoid(self):
        track = Tensor._should_track(self)
        result = Tensor(Sigmoid.forward(self.data), requires_grad=track)
        if track:
            result.context = Function(Sigmoid, self)
        return result

    def softmax(self, dim=None):
        track = Tensor._should_track(self)
        result = Tensor(Softmax.forward(self.data, dim=dim), requires_grad=track)
        if track:
            result.context = Function(Softmax, self, dim)
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
        track = Tensor._should_track(self)
        result = Tensor(CrossEntropyLoss.forward(self.data, targets.data), requires_grad=track)
        if track:
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
        return Tensor(np.random.choice(range(input.shape[1]), size=num_samples, p=input.data[0]))

    @staticmethod
    def cat(tensors, dim=0):
        tensors = [Tensor.ensure_tensor(t) for t in tensors]
        track = Tensor._should_track(*tensors)
        result = Tensor(Cat.forward(*[t.data for t in tensors], dim), requires_grad=track)
        if track:
            result.context = Function(Cat, *tensors, dim)
        return result

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
            raise NotImplementedError("Only 'cpu' device is supported for the Tensor class.")
