from __future__ import annotations

import numpy as np

from punytorch.activations import ReLU, Sigmoid, Softmax
from punytorch.mlops import Reshape
from punytorch.ops import Add, Function, MatMul, Mod, Mul, Pow, Sub, Tanh, TrueDiv


class Tensor:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        self.requires_grad = requires_grad
        # if requires_grad is True, then we need to initialize the gradient to zeros
        # and make sure that they're floats, since backprop uses floats
        self.grad = np.zeros_like(self.data, dtype=np.float64) if requires_grad else None
        self.dtype = self.data.dtype
        self.context = None

    @property
    def shape(self):
        return self.data.shape

    def __str__(self):
        return "tensor([" + ", ".join(f"{x:.1f}" for x in self.data.flatten()) + "])"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # Wrap the sliced array in a Tensor object
        if isinstance(index, Tensor):
            index = index.data
        if isinstance(index, (int, np.integer, slice)):
            # Handle integer, numpy integer, and slice indexing
            return Tensor(self.data[index], requires_grad=self.requires_grad)
        elif isinstance(index, (tuple, list)) or (isinstance(index, np.ndarray) and index.ndim == 1):
            # Handle 1D fancy indexing for numpy arrays
            return Tensor(self.data[index], requires_grad=self.requires_grad)
        elif isinstance(index, np.ndarray) and index.ndim == 2:
            # Handle 2D fancy indexing for numpy arrays
            return Tensor(self.data[index[:, None], index], requires_grad=self.requires_grad)
        else:
            raise IndexError("Indexing with the provided index is not supported")

    @property
    def T(self):
        return Tensor(np.transpose(self.data))

    def transpose(self, dim0, dim1):
        """
        Returns a tensor with dimensions dim0 and dim1 swapped.

        Args:
            dim0 (int): The first dimension to be swapped.
            dim1 (int): The second dimension to be swapped.

        Returns:
            Tensor: A tensor with dimensions dim0 and dim1 swapped.
        """
        if dim0 >= self.data.ndim or dim1 >= self.data.ndim:
            raise ValueError(
                f"Dimension out of range. Tensor has {self.data.ndim} dimensions but dim0={dim0} or dim1={dim1} was provided."
            )

        axes = list(range(self.data.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        data = self.data.transpose(axes)
        return Tensor(data, requires_grad=self.requires_grad)

    def tolist(self):
        return self.data.tolist()

    def item(self):
        assert np.prod(self.data.shape) == 1, "Only one element tensors can be converted to Python scalars"
        return self.data.item()

    def __gt__(self, other):
        if isinstance(other, Tensor):
            return self.data > other.data
        else:
            return self.data > other

    def __lt__(self, other):
        if isinstance(other, Tensor):
            return self.data < other.data
        else:
            return self.data < other

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
                    if isinstance(arg, Tensor) and arg.requires_grad:
                        if arg.grad is None:
                            arg.grad = np.zeros_like(arg.data)
                        arg.grad += grad_arg.data  # Ensure grad_arg is a numpy array
                        stack.append((arg, grad_arg))

    @staticmethod
    def ensure_tensor(data):
        if not isinstance(data, Tensor):
            data = Tensor(np.array(data))
        return data

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
        if isinstance(shape[0], tuple):
            shape = shape[0]
        curr = self.prod(self.shape)
        target = self.prod((s for s in shape if s != -1))
        shape = tuple(curr // target if s == -1 else s for s in shape)
        return Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)

    """
    BINARY OPS
    """

    def __add__(self, other):
        result = Tensor(
            Add.forward(self, other), requires_grad=self.requires_grad or getattr(other, "requires_grad", False)
        )
        if result.requires_grad:
            result.context = Function(Add, self, other)
        return result

    def __sub__(self, other):
        result = Tensor(
            Sub.forward(self, other), requires_grad=self.requires_grad or getattr(other, "requires_grad", False)
        )
        if result.requires_grad:
            result.context = Function(Sub, self, other)
        return result

    def __mul__(self, other):
        result = Tensor(
            Mul.forward(self, other), requires_grad=self.requires_grad or getattr(other, "requires_grad", False)
        )
        if result.requires_grad:
            result.context = Function(Mul, self, other)
        return result

    def __truediv__(self, other):
        result = Tensor(
            TrueDiv.forward(self, other), requires_grad=self.requires_grad or getattr(other, "requires_grad", False)
        )
        if result.requires_grad:
            result.context = Function(TrueDiv, self, other)
        return result

    def __mod__(self, other):
        result = Tensor(
            Mod.forward(self, other), requires_grad=self.requires_grad or getattr(other, "requires_grad", False)
        )
        if result.requires_grad:
            result.context = Function(Mod, self, other)
        return result

    def __pow__(self, other):
        result = Tensor(
            Pow.forward(self, other), requires_grad=self.requires_grad or getattr(other, "requires_grad", False)
        )
        if result.requires_grad:
            result.context = Function(Pow, self, other)
        return result

    def __matmul__(self, other):
        result = Tensor(
            MatMul.forward(self, other), requires_grad=self.requires_grad or getattr(other, "requires_grad", False)
        )
        if result.requires_grad:
            result.context = Function(MatMul, self, other)
        return result

    def __tanh__(self):
        result = Tensor(Tanh.forward(self), requires_grad=self.requires_grad)
        if result.requires_grad:
            result.context = Function(Tanh, self)
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
            raise NotImplementedError("Only 'cpu' device is supported for the Tensor class.")

    def mean(self, axis=None, keepdims=False):
        """
        Compute the mean along the specified axis.
        """
        mean_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(mean_data)
        if self.requires_grad:
            # Assuming the Tensor class has a way to set the grad_fn after construction
            size = self.data.size if axis is None else self.data.shape[axis]

            def grad_fn(grad):
                return grad * np.ones_like(self.data) / size

            result.grad_fn = grad_fn
        return result
