from punytorch.ops import Operation
import numpy as np


class Reshape(Operation):
    @staticmethod
    def forward(x, shape):
        return x.__class__(x.data.reshape(tuple(shape)))

    @staticmethod
    def backward(ctx: Operation, grad):
        x, _ = ctx.args
        return grad.__class__(grad.data.reshape(x.shape)), None


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
        return Tensor(np.transpose(grad_data)),
