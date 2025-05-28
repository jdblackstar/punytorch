import numpy as np


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
        from punytorch.tensor import Tensor

        # For x gradient
        grad_x_data = grad.data
        # Sum over dimensions that were broadcast
        ndims_added = grad_x_data.ndim - x.data.ndim
        for i in range(ndims_added):
            grad_x_data = np.sum(grad_x_data, axis=0)
        # Sum over dimensions that were broadcast from size 1
        for i, (dim, grad_dim) in enumerate(zip(x.data.shape, grad_x_data.shape)):
            if dim == 1 and grad_dim > 1:
                grad_x_data = np.sum(grad_x_data, axis=i, keepdims=True)

        # For y gradient
        grad_y_data = grad.data
        # Sum over dimensions that were broadcast
        ndims_added = grad_y_data.ndim - y.data.ndim
        for i in range(ndims_added):
            grad_y_data = np.sum(grad_y_data, axis=0)
        # Sum over dimensions that were broadcast from size 1
        for i, (dim, grad_dim) in enumerate(zip(y.data.shape, grad_y_data.shape)):
            if dim == 1 and grad_dim > 1:
                grad_y_data = np.sum(grad_y_data, axis=i, keepdims=True)

        return Tensor(grad_x_data), Tensor(grad_y_data)


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
        return grad, -grad


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
        return y * grad, x * grad


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
        grad_x = grad / y
        grad_y = -grad * x / (y * y)
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
        return grad, 0


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
        from punytorch.tensor import Tensor

        x, y = context.args
        grad_x = grad * (y * x ** (y - 1))
        grad_y = grad * (x**y * Tensor(np.log(x.data)))
        return grad_x, grad_y


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
        return grad @ y.T, x.T @ grad


class Tanh(Operation):
    @staticmethod
    def forward(x):
        """
        z = tanh(x)
        """
        return np.tanh(x)

    @staticmethod
    def backward(context, grad):
        """
        d(tanh(x))/dx = 1 - tanh(x)^2

        return (1 - tanh(x)^2) * grad
        """
        from punytorch.tensor import Tensor

        x = context.args[0].data
        grad_data = grad.data if isinstance(grad, Tensor) else grad
        grad_tanh = 1 - np.tanh(x) ** 2
        return (Tensor(grad_tanh * grad_data),)


class Transpose(Operation):
    """
    Implements matrix transpose operation with proper gradient flow.
    """

    @staticmethod
    def forward(x):
        """
        z = x.T (transpose of x)
        """
        return np.transpose(x.data)

    @staticmethod
    def backward(context, grad):
        """
        If Z = X.T, then d(Z)/dX = grad.T
        The gradient simply needs to be transposed back.
        """
        from punytorch.tensor import Tensor

        # Transpose the gradient back
        grad_data = grad.data if isinstance(grad, Tensor) else grad
        return (Tensor(np.transpose(grad_data)),)


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
        from punytorch.tensor import Tensor

        x, _ = context.args
        grad_data = grad.data if hasattr(grad, "data") else grad
        return Tensor(grad_data.reshape(x.shape)), None


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
        from punytorch.tensor import Tensor

        x, axis, keepdims = context.args
        grad_data = grad.data if hasattr(grad, "data") else grad

        # Reshape gradient to match original tensor dimensions if keepdims=False
        if axis is not None and not keepdims:
            # Add back the reduced dimensions
            if isinstance(axis, int):
                axis = (axis,)
            elif axis is None:
                axis = tuple(range(x.data.ndim))
            else:
                axis = tuple(axis)

            # Expand dimensions that were reduced
            for ax in sorted(axis):
                grad_data = np.expand_dims(grad_data, axis=ax)

        # Broadcast gradient to original tensor shape
        grad_output = np.broadcast_to(grad_data, x.data.shape)
        return Tensor(grad_output), None, None


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
        from punytorch.tensor import Tensor

        x, axis, keepdims = context.args
        grad_data = grad.data if hasattr(grad, "data") else grad

        # Calculate the number of elements that contributed to the mean
        if axis is None:
            num_elements = x.data.size
        else:
            if isinstance(axis, int):
                num_elements = x.data.shape[axis]
            else:
                num_elements = np.prod([x.data.shape[ax] for ax in axis])

        # Reshape gradient to match original tensor dimensions if keepdims=False
        if axis is not None and not keepdims:
            if isinstance(axis, int):
                axis = (axis,)
            elif axis is None:
                axis = tuple(range(x.data.ndim))
            else:
                axis = tuple(axis)

            # Expand dimensions that were reduced
            for ax in sorted(axis):
                grad_data = np.expand_dims(grad_data, axis=ax)

        # Broadcast gradient to original tensor shape and scale by 1/N
        grad_output = np.broadcast_to(grad_data, x.data.shape) / num_elements
        return Tensor(grad_output), None, None


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
        from punytorch.tensor import Tensor

        x, axis, keepdims = context.args
        grad_data = grad.data if hasattr(grad, "data") else grad

        # Find the maximum values and create a mask
        max_vals = np.max(x.data, axis=axis, keepdims=True)
        max_mask = (x.data == max_vals).astype(np.float64)

        # Reshape gradient to match original tensor dimensions if keepdims=False
        if axis is not None and not keepdims:
            if isinstance(axis, int):
                axis = (axis,)
            elif axis is None:
                axis = tuple(range(x.data.ndim))
            else:
                axis = tuple(axis)

            # Expand dimensions that were reduced
            for ax in sorted(axis):
                grad_data = np.expand_dims(grad_data, axis=ax)

        # Broadcast gradient and apply mask (only max elements get gradient)
        grad_broadcast = np.broadcast_to(grad_data, x.data.shape)

        # Handle case where multiple elements achieve the maximum (split gradient)
        num_max_elements = np.sum(max_mask, axis=axis, keepdims=True)
        num_max_elements = np.where(
            num_max_elements == 0, 1, num_max_elements
        )  # Avoid division by zero

        grad_output = grad_broadcast * max_mask / num_max_elements
        return Tensor(grad_output), None, None
