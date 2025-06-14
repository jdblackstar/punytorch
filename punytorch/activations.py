import numpy as np
from punytorch.ops import Operation


class ReLU(Operation):
    """
    Implements the ReLU (Rectified Linear Unit) activation function for a neural network.
    """

    @staticmethod
    def forward(x):
        """
        Performs the forward pass of the ReLU activation function.

        Args:
            x (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The output after applying the ReLU function, which is max(0, x).
        """
        return np.maximum(0, x.data)

    @staticmethod
    def backward(context, grad):
        """
        Performs the backward pass of the ReLU activation function for backpropagation.

        Args:
            context (Context): An object storing intermediate values from the forward pass.
            grad (numpy.ndarray or scalar): The gradient of the loss with respect to the output.

        Returns:
            tuple: A tuple where the first element is the gradient of the loss with respect to the input,
                   and the second element is None (since ReLU has no parameters to update).
        """
        from punytorch.tensor import Tensor

        x = context.args[0].data
        # grad wasn't broadcasting to the same shape as x, so:
        grad_data = grad.data if isinstance(grad, Tensor) else grad
        grad_data = np.ones_like(x) if np.isscalar(grad_data) else grad_data
        return Tensor((x > 0).astype(np.float64) * grad_data), None


class Sigmoid(Operation):
    """
    Implements the Sigmoid activation function for a neural network.
    """

    @staticmethod
    def forward(x):
        """
        Performs the forward pass of the Sigmoid activation function.

        Args:
            x (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The output after applying the Sigmoid function, which is 1 / (1 + exp(-x)).
        """
        return 1 / (1 + np.exp(-x.data))

    @staticmethod
    def backward(context, grad):
        """
        Performs the backward pass of the Sigmoid activation function for backpropagation.

        Args:
            context (Context): An object storing intermediate values from the forward pass.
            grad (numpy.ndarray or scalar): The gradient of the loss with respect to the output.

        Returns:
            tuple: A tuple where the first element is the gradient of the loss with respect to the input,
                   and the second element is None (since Sigmoid has no parameters to update).
        """
        from punytorch.tensor import Tensor

        x = context.args[0].data
        sigmoid_x = 1 / (1 + np.exp(-x))
        grad_data = grad.data if isinstance(grad, Tensor) else grad
        return Tensor(sigmoid_x * (1 - sigmoid_x) * grad_data), None


class Softmax(Operation):
    """
    Implements the Softmax activation function for a neural network.
    """

    @staticmethod
    def forward(x, dim=None):
        """
        Performs the forward pass of the Softmax activation function.

        Args:
            x (numpy.ndarray): The input data.
            dim (int, optional): The dimension along which to apply the softmax function. If None, apply softmax along the last dimension.

        Returns:
            numpy.ndarray: The output after applying the Softmax function, which is exp(x) / sum(exp(x)).
                        The max(x) is subtracted for numerical stability.
        """
        if dim is None:
            dim = -1
        input_data = x.data if hasattr(x, "data") else x
        e_x = np.exp(
            input_data - np.max(input_data, axis=dim, keepdims=True)
        )  # subtract max(x) for numerical stability
        return e_x / np.sum(e_x, axis=dim, keepdims=True)

    @staticmethod
    def backward(context, grad):
        """
        Performs the backward pass of the Softmax activation function for backpropagation.

        The gradient of softmax is a bit complex to compute, it's not as straightforward as other functions.
        We need to compute the Jacobian matrix of the softmax function and then multiply with the gradient.

        Args:
            context (Context): An object storing intermediate values from the forward pass.
                               In this case, it contains the input data to the activation function.
                               This data is needed to compute the gradient during the backward pass.
            grad (numpy.ndarray or scalar): The gradient of the loss with respect to the output.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input.
        """
        from punytorch.tensor import Tensor

        x = context.args[0].data
        softmax_x = np.exp(x - np.max(x))
        softmax_x /= np.sum(softmax_x, axis=0)
        grad_data = grad.data if isinstance(grad, Tensor) else grad
        grad_data = np.ones_like(x) if np.isscalar(grad_data) else grad_data
        grad_input = np.zeros_like(x)
        for i in range(len(x)):
            for j in range(len(x)):
                if i == j:
                    grad_input[i] += grad_data[j] * softmax_x[i] * (1 - softmax_x[j])
                else:
                    grad_input[i] -= grad_data[j] * softmax_x[i] * softmax_x[j]
        return Tensor(grad_input), None
