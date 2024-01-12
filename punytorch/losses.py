import numpy as np

from punytorch.tensor import Tensor
from punytorch.activations import Softmax


class MSELoss:
    @staticmethod
    def forward(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def backward(y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.size


class CrossEntropyLoss:
    @staticmethod
    def forward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        probs = Softmax.forward(y_pred)
        log_likelihood = -np.log(probs) * y_true  # Element-wise multiplication
        loss = np.sum(log_likelihood) / len(y_true)
        return Tensor(loss, requires_grad=True)

    @staticmethod
    def backward(y_pred, y_true):
        exps = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True)) + 1e-22
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        d_loss = probs - y_true  # Subtract the one-hot encoded labels
        d_loss /= len(y_true)
        return Tensor(d_loss)  # Wrap the result in a Tensor


class BinaryCrossEntropyLoss:
    @staticmethod
    def forward(y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def backward(y_pred, y_true):
        return (y_pred - y_true) / (y_pred * (1 - y_pred))


class CategoricalCrossEntropyLoss:
    @staticmethod
    def forward(y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred))

    @staticmethod
    def backward(y_pred, y_true):
        return -y_true / y_pred
