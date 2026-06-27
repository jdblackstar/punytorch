import numpy as np

from punytorch.ops import Operation


class MSELoss(Operation):
    """
    Implements the Mean Squared Error (MSE) loss function for a neural network.
    """

    @staticmethod
    def forward(y_pred, y_true) -> float:
        """
        Computes the forward pass of the MSE loss function.

        Args:
            y_pred (Tensor): The predicted values.
            y_true (Tensor): The true values.

        Returns:
            float: The MSE loss, which is the mean of the squared differences between the predicted and true values.
        """
        loss = np.mean((y_pred - y_true) ** 2)
        return loss

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
        grad_pred = 2 * (y_pred.data - y_true.data) / y_pred.data.size
        grad_data = np.asarray(grad, dtype=np.float64)
        return grad_pred * grad_data, None


class CrossEntropyLoss(Operation):
    """
    Combines LogSoftmax and NLLLoss in a single operation.
    Expects raw logits as input, similar to PyTorch's nn.CrossEntropyLoss.
    """

    @staticmethod
    def forward(logits, targets) -> float:
        """
        Computes cross entropy loss from raw logits.

        Args:
            logits (Tensor): Raw model outputs, shape (batch_size, num_classes)
            targets (Tensor): One-hot encoded labels, shape (batch_size, num_classes)

        Returns:
            float: Scalar loss value
        """
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
            targets = targets.reshape(1, -1)

        # Numerical stability: Subtract max logit (prevents exp overflow)
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)

        # Log-softmax implementation
        exp_logits = np.exp(shifted_logits)
        log_sum_exp = np.log(np.sum(exp_logits, axis=1, keepdims=True))
        log_softmax = shifted_logits - log_sum_exp

        # Cross entropy is negative sum of true_probs * log_probs
        batch_losses = -np.sum(targets * log_softmax, axis=1)
        loss = np.mean(batch_losses)

        return loss

    @staticmethod
    def backward(context, grad) -> tuple:
        """
        Computes gradient of loss with respect to logits.

        Args:
            context: Function context containing logits and targets
            grad: Gradient from upstream (usually 1.0 for loss)

        Returns:
            tuple: (Gradient with respect to logits, None for targets)
        """
        logits, targets = context.args

        logits_data = logits.data
        targets_data = targets.data
        original_shape = logits_data.shape
        if logits_data.ndim == 1:
            logits_data = logits_data.reshape(1, -1)
            targets_data = targets_data.reshape(1, -1)

        # Compute softmax (reuse the stability trick)
        shifted_logits = logits_data - np.max(logits_data, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Gradient is (softmax - targets) / batch_size
        batch_size = logits_data.shape[0]
        grad_logits = (softmax - targets_data) / batch_size
        grad_logits = grad_logits.reshape(original_shape)

        # Scale by upstream gradient and return as Tensor
        return grad_logits * np.asarray(grad, dtype=np.float64), None


class BinaryCrossEntropyLoss(Operation):
    """
    Implements the Binary Cross Entropy loss function for a neural network.
    """

    @staticmethod
    def forward(y_pred, y_true) -> float:
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
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
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
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred.data, epsilon, 1 - epsilon)
        grad_pred = (y_pred_clipped - y_true.data) / (y_pred_clipped * (1 - y_pred_clipped))
        grad_data = np.asarray(grad, dtype=np.float64)
        return grad_pred * grad_data / y_pred.data.size, None


class CategoricalCrossEntropyLoss(Operation):
    """
    Implements the Categorical Cross Entropy loss function for a neural network.
    """

    @staticmethod
    def forward(y_pred, y_true) -> float:
        """
        Computes the forward pass of the Categorical Cross Entropy loss function.

        Args:
            y_pred (Tensor): The predicted values.
            y_true (Tensor): The true values.

        Returns:
            float: The Categorical Cross Entropy loss.
        """
        # Add small epsilon for numerical stability
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred_clipped))
        return loss

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
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred.data, epsilon, 1 - epsilon)
        grad_pred = -y_true.data / y_pred_clipped
        grad_data = np.asarray(grad, dtype=np.float64)
        return grad_pred * grad_data, None
