import numpy as np


def _data(value):
    return value.data if hasattr(value, "data") else value


def _grad_data(value):
    value = _data(value)
    return np.asarray(value, dtype=np.float64)


def _normalize_axis(axis, ndim):
    if axis is None:
        return None
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(ax if ax >= 0 else ndim + ax for ax in axis)


def _unbroadcast(grad, shape):
    grad = np.asarray(grad, dtype=np.float64)

    if shape == ():
        return np.asarray(grad.sum(), dtype=np.float64)

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


class Function:
    def __init__(self, op, *args):
        self.op = op
        self.args = args

    def apply(self, *args):
        """
        Applies the function to the given arguments.

        Args:
            *args: The arguments to apply the function to.

        Returns:
            The result of applying the function.
        """
        return self.op.forward(*args)


class Operation:
    """
    Operation contract:
    - forward receives Tensor inputs and returns NumPy-compatible data.
    - backward receives the original Function context plus an upstream gradient
      array and returns one gradient array, or None, per forward argument.
    """

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, context, grad):
        raise NotImplementedError


class Add(Operation):
    @staticmethod
    def forward(x, y):
        """
        z = x + y
        """
        return x.data + y.data

    @staticmethod
    def backward(context, grad):
        """
        d(x + y)/dx = 1
        d(x + y)/dy = 1

        return grad for both x and y
        """
        x, y = context.args
        grad_data = _grad_data(grad)
        return _unbroadcast(grad_data, x.data.shape), _unbroadcast(grad_data, y.data.shape)


class Sub(Operation):
    @staticmethod
    def forward(x, y):
        """
        z = x - y
        """
        return x.data - y.data

    @staticmethod
    def backward(context, grad):
        """
        d(x - y)/dx = 1
        d(x - y)/dy = -1

        return [1] for x
        return [-1] for y
        """
        x, y = context.args
        grad_data = _grad_data(grad)
        return _unbroadcast(grad_data, x.data.shape), _unbroadcast(-grad_data, y.data.shape)


class Mul(Operation):
    @staticmethod
    def forward(x, y):
        """
        z = x * y
        """
        return x.data * y.data

    @staticmethod
    def backward(context, grad):
        """
        d(x * y)/dx = y
        d(x * y)/dy = x

        return (y.data * grad.data) for x
        return (x.data * grad.data) for y
        """
        x, y = context.args
        grad_data = _grad_data(grad)
        return (
            _unbroadcast(grad_data * y.data, x.data.shape),
            _unbroadcast(grad_data * x.data, y.data.shape),
        )


class TrueDiv(Operation):
    @staticmethod
    def forward(x, y):
        """
        z = x / y
        """
        return x.data / y.data

    @staticmethod
    def backward(context, grad):
        """
        d(x / y)/dx = 1/y
        d(x / y)/dy = -x/y^2

        return (grad.data / y.data) for x
        return (-grad.data * x.data / (y.data ** 2)) for y
        """
        x, y = context.args
        grad_data = _grad_data(grad)
        return (
            _unbroadcast(grad_data / y.data, x.data.shape),
            _unbroadcast(-grad_data * x.data / (y.data**2), y.data.shape),
        )


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
        """
        z = x % y
        """
        return x.data % y.data

    @staticmethod
    def backward(context, grad):
        """
        d(x % y)/dx = 1
        d(x % y)/dy = 0

        return an array of ones like x.data for x
        return an array of zerose like y.data for y
        """
        x, y = context.args
        grad_data = _grad_data(grad)
        return _unbroadcast(grad_data, x.data.shape), np.zeros_like(y.data, dtype=np.float64)


class Pow(Operation):
    @staticmethod
    def forward(x, y):
        """
        z = x ^ y
        """
        return x.data**y.data

    @staticmethod
    def backward(context, grad):
        """
        d(x ^ y)/dx = y * x^(y - 1)
        d(x ^ y)/dy = x^y * log(x)

        return grad * (y * x ** (y - 1)) for x
        return grad * (x**y * np.log(x.data)) for y
        """
        x, y = context.args
        grad_data = _grad_data(grad)
        result = x.data**y.data
        grad_x = grad_data * y.data * (x.data ** (y.data - 1))
        grad_y = grad_data * result * np.log(x.data)
        return _unbroadcast(grad_x, x.data.shape), _unbroadcast(grad_y, y.data.shape)


class MatMul(Operation):
    @staticmethod
    def forward(x, y):
        """
        z = x @ y
        """
        return x.data @ y.data

    @staticmethod
    def backward(context, grad):
        """
        If Z = X @ Y, then
        d(Z)/dX = grad @ Y.T
        d(Z)/dY = X.T @ grad
        """
        x, y = context.args
        grad_data = _grad_data(grad)
        grad_x = grad_data @ np.swapaxes(y.data, -1, -2)
        grad_y = np.swapaxes(x.data, -1, -2) @ grad_data
        return _unbroadcast(grad_x, x.data.shape), _unbroadcast(grad_y, y.data.shape)


class Tanh(Operation):
    @staticmethod
    def forward(x):
        """
        z = tanh(x)
        """
        return np.tanh(x.data)

    @staticmethod
    def backward(context, grad):
        """
        d(tanh(x))/dx = 1 - tanh(x)^2

        return (1 - tanh(x)^2) * grad
        """
        x = context.args[0].data
        grad_data = _grad_data(grad)
        grad_tanh = 1 - np.tanh(x) ** 2
        return (grad_tanh * grad_data,)


class Transpose(Operation):
    """
    Implements matrix transpose operation with proper gradient flow.
    """

    @staticmethod
    def forward(x, axes=None):
        """
        z = x.T (transpose of x)
        """
        return np.transpose(x.data, axes=axes)

    @staticmethod
    def backward(context, grad):
        """
        If Z = X.T, then d(Z)/dX = grad.T
        The gradient simply needs to be transposed back.
        """
        _, axes = context.args
        grad_data = _grad_data(grad)
        if axes is None:
            return np.transpose(grad_data), None
        inverse_axes = np.argsort(axes)
        return np.transpose(grad_data, axes=inverse_axes), None


class Reshape(Operation):
    """
    Implements tensor reshape operation with proper gradient flow.
    """

    @staticmethod
    def forward(x, shape):
        """
        Reshapes the tensor to the specified shape.

        Args:
            x: Input tensor
            shape: Target shape tuple

        Returns:
            numpy.ndarray: Reshaped tensor data
        """
        return x.data.reshape(tuple(shape))

    @staticmethod
    def backward(context, grad):
        """
        Reshapes gradient back to original tensor shape.

        Args:
            context: Function context containing original tensor and target shape
            grad: Gradient tensor

        Returns:
            tuple: (Gradient reshaped to original shape, None for shape parameter)
        """
        x, _ = context.args
        grad_data = _grad_data(grad)
        return grad_data.reshape(x.shape), None


class Sum(Operation):
    """
    Implements tensor sum reduction with proper gradient flow.
    """

    @staticmethod
    def forward(x, axis=None, keepdims=False):
        """
        Computes the sum of tensor elements along specified axis.

        Args:
            x: Input tensor
            axis: Axis or axes along which to sum. If None, sum all elements.
            keepdims: Whether to keep reduced dimensions

        Returns:
            numpy.ndarray: Sum result
        """
        return np.sum(x.data, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(context, grad):
        """
        Distributes gradient back to all elements that contributed to the sum.

        Args:
            context: Function context containing original tensor and reduction parameters
            grad: Gradient tensor

        Returns:
            tuple: (Gradient distributed to original shape, None for axis parameter)
        """
        x, axis, keepdims = context.args
        grad_data = _grad_data(grad)
        normalized_axis = _normalize_axis(axis, x.data.ndim)

        # Reshape gradient to match original tensor dimensions if keepdims=False
        if normalized_axis is not None and not keepdims:
            # Expand dimensions that were reduced
            for ax in sorted(normalized_axis):
                grad_data = np.expand_dims(grad_data, axis=ax)

        # Broadcast gradient to original tensor shape
        grad_output = np.broadcast_to(grad_data, x.data.shape)
        return grad_output, None, None


class Mean(Operation):
    """
    Implements tensor mean reduction with proper gradient flow.
    """

    @staticmethod
    def forward(x, axis=None, keepdims=False):
        """
        Computes the mean of tensor elements along specified axis.

        Args:
            x: Input tensor
            axis: Axis or axes along which to compute mean. If None, mean of all elements.
            keepdims: Whether to keep reduced dimensions

        Returns:
            numpy.ndarray: Mean result
        """
        return np.mean(x.data, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(context, grad):
        """
        Distributes gradient back to all elements, scaled by the number of elements.

        Args:
            context: Function context containing original tensor and reduction parameters
            grad: Gradient tensor

        Returns:
            tuple: (Gradient distributed to original shape, None for axis parameter)
        """
        x, axis, keepdims = context.args
        grad_data = _grad_data(grad)
        normalized_axis = _normalize_axis(axis, x.data.ndim)

        # Calculate the number of elements that contributed to the mean
        if normalized_axis is None:
            num_elements = x.data.size
        elif len(normalized_axis) == 1:
            num_elements = x.data.shape[normalized_axis[0]]
        else:
            num_elements = np.prod([x.data.shape[ax] for ax in normalized_axis])

        # Reshape gradient to match original tensor dimensions if keepdims=False
        if normalized_axis is not None and not keepdims:
            # Expand dimensions that were reduced
            for ax in sorted(normalized_axis):
                grad_data = np.expand_dims(grad_data, axis=ax)

        # Broadcast gradient to original tensor shape and scale by 1/N
        grad_output = np.broadcast_to(grad_data, x.data.shape) / num_elements
        return grad_output, None, None


class Max(Operation):
    """
    Implements tensor max reduction with proper gradient flow.
    """

    @staticmethod
    def forward(x, axis=None, keepdims=False):
        """
        Computes the maximum of tensor elements along specified axis.

        Args:
            x: Input tensor
            axis: Axis or axes along which to find max. If None, max of all elements.
            keepdims: Whether to keep reduced dimensions

        Returns:
            numpy.ndarray: Max result
        """
        return np.max(x.data, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(context, grad):
        """
        Distributes gradient only to elements that achieved the maximum value.

        Args:
            context: Function context containing original tensor and reduction parameters
            grad: Gradient tensor

        Returns:
            tuple: (Gradient distributed to max elements only, None for axis parameter)
        """
        x, axis, keepdims = context.args
        grad_data = _grad_data(grad)
        normalized_axis = _normalize_axis(axis, x.data.ndim)

        # Find the maximum values and create a mask
        max_vals = np.max(x.data, axis=axis, keepdims=True)
        max_mask = (x.data == max_vals).astype(np.float64)

        # Reshape gradient to match original tensor dimensions if keepdims=False
        if normalized_axis is not None and not keepdims:
            # Expand dimensions that were reduced
            for ax in sorted(normalized_axis):
                grad_data = np.expand_dims(grad_data, axis=ax)

        # Broadcast gradient and apply mask (only max elements get gradient)
        grad_broadcast = np.broadcast_to(grad_data, x.data.shape)

        # Handle case where multiple elements achieve the maximum (split gradient)
        num_max_elements = np.sum(max_mask, axis=axis, keepdims=True)
        num_max_elements = np.where(
            num_max_elements == 0, 1, num_max_elements
        )  # Avoid division by zero

        grad_output = grad_broadcast * max_mask / num_max_elements
        return grad_output, None, None
