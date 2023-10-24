import numpy as np


class MSELoss:
    @staticmethod
    def forward(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def backward(y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.size


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
