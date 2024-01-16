import numpy as np

from punytorch.tensor import Tensor
from punytorch.activations import Softmax


class MSELoss:
    """
    Implements the Mean Squared Error (MSE) loss function for a neural network.
    """

    @staticmethod
    def forward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Computes the forward pass of the MSE loss function.

        Args:
            y_pred (numpy.ndarray): The predicted values.
            y_true (numpy.ndarray): The true values.

        Returns:
            Tensor: The MSE loss, which is the mean of the squared differences between the predicted and true values.
                    The result is wrapped in a Tensor.
        """
        loss = np.mean((y_pred - y_true) ** 2)
        return Tensor(loss, requires_grad=True)

    @staticmethod
    def backward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Computes the backward pass of the MSE loss function for backpropagation.

        Args:
            y_pred (numpy.ndarray): The predicted values.
            y_true (numpy.ndarray): The true values.

        Returns:
            Tensor: The gradient of the MSE loss with respect to the predicted values.
                    This is computed as twice the mean difference between the predicted and true values.
                    The result is wrapped in a Tensor.
        """
        grad = 2 * (y_pred - y_true) / y_pred.size
        return Tensor(grad)


class CrossEntropyLoss:
    """
    Implements the Cross Entropy loss function for a neural network.
    """

    @staticmethod
    def forward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Computes the forward pass of the Cross Entropy loss function.

        Args:
            y_pred (Tensor): The predicted values.
            y_true (Tensor): The true values.

        Returns:
            Tensor: The Cross Entropy loss, which is the negative log likelihood of the true labels given the predictions.
                    The result is wrapped in a Tensor.
        """
        probs = Softmax.forward(y_pred)
        log_likelihood = -np.log(probs) * y_true  # Element-wise multiplication
        loss = np.sum(log_likelihood) / len(y_true)
        return Tensor(loss, requires_grad=True)

    @staticmethod
    def backward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Computes the backward pass of the Cross Entropy loss function for backpropagation.

        Args:
            y_pred (Tensor): The predicted values.
            y_true (Tensor): The true values.

        Returns:
            Tensor: The gradient of the Cross Entropy loss with respect to the predicted values.
                    This is computed as the difference between the softmax probabilities and the true labels.
                    The result is wrapped in a Tensor.
        """
        exps = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True)) + 1e-22
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        d_loss = probs - y_true  # Subtract the one-hot encoded labels
        d_loss /= len(y_true)
        return Tensor(d_loss)  # Wrap the result in a Tensor


class BinaryCrossEntropyLoss:
    """
    Implements the Binary Cross Entropy loss function for a neural network.
    """

    @staticmethod
    def forward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Computes the forward pass of the Binary Cross Entropy loss function.

        Args:
            y_pred (Tensor): The predicted values.
            y_true (Tensor): The true values.

        Returns:
            Tensor: The Binary Cross Entropy loss, which is the negative mean of (y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)).
                    The result is wrapped in a Tensor.
        """
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return Tensor(loss, requires_grad=True)

    @staticmethod
    def backward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Computes the backward pass of the Binary Cross Entropy loss function for backpropagation.

        Args:
            y_pred (Tensor): The predicted values.
            y_true (Tensor): The true values.

        Returns:
            Tensor: The gradient of the Binary Cross Entropy loss with respect to the predicted values.
                    This is computed as (y_pred - y_true) / (y_pred * (1 - y_pred)).
                    The result is wrapped in a Tensor.
        """
        grad = (y_pred - y_true) / (y_pred * (1 - y_pred))
        return Tensor(grad)


class CategoricalCrossEntropyLoss:
    """
    Implements the Categorical Cross Entropy loss function for a neural network.
    """

    @staticmethod
    def forward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Computes the forward pass of the Categorical Cross Entropy loss function.

        Args:
            y_pred (Tensor): The predicted values.
            y_true (Tensor): The true values.

        Returns:
            Tensor: The Categorical Cross Entropy loss, which is the negative sum of y_true * log(y_pred).
                    The result is wrapped in a Tensor.
        """
        loss = -np.sum(y_true * np.log(y_pred))
        return Tensor(loss, requires_grad=True)

    @staticmethod
    def backward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Computes the backward pass of the Categorical Cross Entropy loss function for backpropagation.

        Args:
            y_pred (Tensor): The predicted values.
            y_true (Tensor): The true values.

        Returns:
            Tensor: The gradient of the Categorical Cross Entropy loss with respect to the predicted values.
                    This is computed as -y_true / y_pred.
                    The result is wrapped in a Tensor.
        """
        grad = -y_true / y_pred
        return Tensor(grad)
