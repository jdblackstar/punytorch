import numpy as np

from punytorch.tensor import Tensor


class MSELoss:
    @staticmethod
    def forward(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def backward(y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.size


class CrossEntropyLoss:
    @staticmethod
    def forward(y_pred, y_true):
        exps = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True) + 1e-22
        log_likelihood = -np.log(probs[np.arange(len(y_true)), y_true.astype(int)])
        return Tensor(np.mean(log_likelihood))  # Wrap the result in a Tensor

    @staticmethod
    def backward(y_pred, y_true):
        exps = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True)) + 1e-22
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        d_loss = np.zeros_like(probs)
        d_loss[np.arange(len(y_true)), y_true.astype(int)] -= 1
        d_loss += probs
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
