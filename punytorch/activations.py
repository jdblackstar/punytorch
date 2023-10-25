import numpy as np


class ReLU:
    @staticmethod
    def forward(x):
        """
        z = max(0, x)
        """
        return np.maximum(0, x)

    def backward(context, grad):
        """
        d(max(0, x))/dx = 1 if x > 0 else 0
        """
        x = context.args[0].data
        # grad wasn't broadcasting to the same shape as x, so:
        grad = np.ones_like(x) if np.isscalar(grad) else grad
        return (x > 0) * grad, None


class Sigmoid:
    @staticmethod
    def forward(x):
        """
        z = 1 / (1 + exp(-x))
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(context, grad):
        """
        d(1 / (1 + exp(-x)))/dx = f(x) * (1 - f(x))
        """
        x = context.args[0].data
        sigmoid_x = 1 / (1 + np.exp(-x))
        return sigmoid_x * (1 - sigmoid_x) * grad, None


class Softmax:
    @staticmethod
    def forward(x):
        """
        z = exp(x) / sum(exp(x))
        """
        e_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
        return e_x / np.sum(e_x, axis=0)

    @staticmethod
    def backward(context, grad):
        """
        The gradient of softmax is a bit complex to compute, it's not as straightforward as other functions.

        we need to compute the jacobian matrix of the softmax function and then multiply with the gradient.
        """
        x = context.args[0].data
        softmax_x = np.exp(x - np.max(x))
        softmax_x /= np.sum(softmax_x, axis=0)
        grad = np.ones_like(x) if np.isscalar(grad) else grad
        grad_input = np.zeros_like(x)
        for i in range(len(x)):
            for j in range(len(x)):
                if i == j:
                    grad_input[i] += grad[j] * softmax_x[i] * (1 - softmax_x[j])
                else:
                    grad_input[i] -= grad[j] * softmax_x[i] * softmax_x[j]
        return grad_input
