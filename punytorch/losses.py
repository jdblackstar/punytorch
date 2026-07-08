import numpy as np

from punytorch.tensor import Tensor
from punytorch.ops import Operation


def _as_array(value):
    if isinstance(value, Tensor):
        return value.data
    return np.asarray(value)


def _grad_data(grad):
    return grad.data if isinstance(grad, Tensor) else np.asarray(grad)


def _classification_arrays(pred, target):
    pred_array = _as_array(pred)
    target_array = _as_array(target)
    original_shape = pred_array.shape

    if pred_array.ndim == 1:
        pred_array = pred_array.reshape(1, -1)
    elif pred_array.ndim != 2:
        raise ValueError("Expected a 1D or 2D prediction array")

    if target_array.ndim == 1:
        target_array = target_array.reshape(1, -1)
    elif target_array.ndim != 2:
        raise ValueError("Expected a 1D or 2D target array")

    if pred_array.shape != target_array.shape:
        raise ValueError("Predictions and targets must have the same shape")

    return pred_array, target_array, original_shape


class MSELoss(Operation):
    """
    Implements the Mean Squared Error (MSE) loss function for a neural network.
    """

    @staticmethod
    def forward(y_pred, y_true):
        """
        Computes the forward pass of the MSE loss function.

        Args:
            y_pred: The predicted values.
            y_true: The true values.

        Returns:
            float: The MSE loss, which is the mean of the squared differences between the predicted and true values.
        """
        y_pred_array = _as_array(y_pred)
        y_true_array = _as_array(y_true)
        return np.mean((y_pred_array - y_true_array) ** 2)

    @staticmethod
    def backward(context, grad):
        """
        Computes the backward pass of the MSE loss function for backpropagation.

        Args:
            context: Function context containing y_pred and y_true
            grad: Gradient from upstream

        Returns:
            tuple: Gradients with respect to y_pred and y_true
        """
        y_pred, y_true = context.args
        y_pred_array = _as_array(y_pred)
        y_true_array = _as_array(y_true)
        grad_pred = 2 * (y_pred_array - y_true_array) / y_pred_array.size
        grad_data = _grad_data(grad)
        return Tensor(grad_pred * grad_data), None


class CrossEntropyLoss(Operation):
    """
    Combines LogSoftmax and NLLLoss in a single operation.
    Expects raw logits as input, similar to PyTorch's nn.CrossEntropyLoss.
    """

    @staticmethod
    def forward(logits, targets):
        """
        Computes cross entropy loss from raw logits.

        Args:
            logits (Tensor): Raw model outputs, shape (batch_size, num_classes)
            targets (Tensor): One-hot encoded labels, shape (batch_size, num_classes)

        Returns:
            float: Scalar loss value
        """
        logits_array, targets_array, _ = _classification_arrays(logits, targets)

        # Numerical stability: Subtract max logit (prevents exp overflow)
        shifted_logits = logits_array - np.max(logits_array, axis=1, keepdims=True)

        # Log-softmax implementation
        exp_logits = np.exp(shifted_logits)
        log_sum_exp = np.log(np.sum(exp_logits, axis=1, keepdims=True))
        log_softmax = shifted_logits - log_sum_exp

        # Cross entropy is negative sum of true_probs * log_probs
        batch_losses = -np.sum(targets_array * log_softmax, axis=1)
        return np.mean(batch_losses)

    @staticmethod
    def backward(context, grad: Tensor) -> tuple:
        """
        Computes gradient of loss with respect to logits.

        Args:
            context: Function context containing logits and targets
            grad: Gradient from upstream (usually 1.0 for loss)

        Returns:
            tuple: (Gradient with respect to logits, None for targets)
        """
        logits, targets = context.args
        logits_array, targets_array, original_shape = _classification_arrays(logits, targets)

        # Compute softmax (reuse the stability trick)
        shifted_logits = logits_array - np.max(logits_array, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Gradient is (softmax - targets) / batch_size
        batch_size = logits_array.shape[0]
        grad_logits = (softmax - targets_array) / batch_size
        grad_logits = grad_logits.reshape(original_shape)
        grad_data = _grad_data(grad)

        # Scale by upstream gradient and return as Tensor
        return Tensor(grad_logits * grad_data), None


class BinaryCrossEntropyLoss(Operation):
    """
    Implements the Binary Cross Entropy loss function for a neural network.
    """

    @staticmethod
    def forward(y_pred, y_true):
        """
        Computes the forward pass of the Binary Cross Entropy loss function.

        Args:
            y_pred (Tensor): The predicted values.
            y_true (Tensor): The true values.

        Returns:
            float: The Binary Cross Entropy loss.
        """
        # Add small epsilon for numerical stability
        epsilon = 1e-15
        y_pred_array = _as_array(y_pred)
        y_true_array = _as_array(y_true)
        y_pred_clipped = np.clip(y_pred_array, epsilon, 1 - epsilon)
        loss = -np.mean(
            y_true_array * np.log(y_pred_clipped)
            + (1 - y_true_array) * np.log(1 - y_pred_clipped)
        )
        return loss

    @staticmethod
    def backward(context, grad):
        """
        Computes the backward pass of the Binary Cross Entropy loss function for backpropagation.

        Args:
            context: Function context containing y_pred and y_true
            grad: Gradient from upstream

        Returns:
            tuple: Gradients with respect to y_pred and y_true
        """
        y_pred, y_true = context.args
        y_pred_array = _as_array(y_pred)
        y_true_array = _as_array(y_true)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred_array, epsilon, 1 - epsilon)
        grad_pred = (y_pred_clipped - y_true_array) / (
            y_pred_clipped * (1 - y_pred_clipped)
        )
        grad_data = _grad_data(grad)
        return Tensor(grad_pred * grad_data / y_pred_array.size), None


class CategoricalCrossEntropyLoss(Operation):
    """
    Implements the Categorical Cross Entropy loss function for a neural network.
    """

    @staticmethod
    def forward(y_pred, y_true):
        """
        Computes the forward pass of the Categorical Cross Entropy loss function.

        Args:
            y_pred (Tensor): The predicted values.
            y_true (Tensor): The true values.

        Returns:
            float: The Categorical Cross Entropy loss.
        """
        # Add small epsilon for numerical stability
        y_pred_array, y_true_array, _ = _classification_arrays(y_pred, y_true)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred_array, epsilon, 1 - epsilon)
        batch_losses = -np.sum(y_true_array * np.log(y_pred_clipped), axis=1)
        return np.mean(batch_losses)

    @staticmethod
    def backward(context, grad):
        """
        Computes the backward pass of the Categorical Cross Entropy loss function for backpropagation.

        Args:
            context: Function context containing y_pred and y_true
            grad: Gradient from upstream

        Returns:
            tuple: Gradients with respect to y_pred and y_true
        """
        y_pred, y_true = context.args
        y_pred_array, y_true_array, original_shape = _classification_arrays(y_pred, y_true)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred_array, epsilon, 1 - epsilon)
        grad_pred = -y_true_array / y_pred_clipped / y_pred_array.shape[0]
        grad_pred = grad_pred.reshape(original_shape)
        grad_data = _grad_data(grad)
        return Tensor(grad_pred * grad_data), None
