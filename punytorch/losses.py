import numpy as np

from punytorch.tensor import Tensor
from punytorch.ops import Operation


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


class CrossEntropyLoss(Operation):
    """
    Combines LogSoftmax and NLLLoss in a single operation.
    Expects raw logits as input, similar to PyTorch's nn.CrossEntropyLoss.
    """
    
    @staticmethod
    def forward(logits: Tensor, targets: Tensor) -> float:
        """
        Computes cross entropy loss from raw logits.
        
        Args:
            logits (Tensor): Raw model outputs, shape (batch_size, num_classes)
            targets (Tensor): One-hot encoded labels, shape (batch_size, num_classes)
            
        Returns:
            float: Scalar loss value
        """
        # Numerical stability: Subtract max logit (prevents exp overflow)
        shifted_logits = logits.data - np.max(logits.data, axis=1, keepdims=True)
        
        # Log-softmax implementation
        exp_logits = np.exp(shifted_logits)
        log_sum_exp = np.log(np.sum(exp_logits, axis=1, keepdims=True))
        log_softmax = shifted_logits - log_sum_exp
        
        # Cross entropy is negative sum of true_probs * log_probs
        batch_losses = -np.sum(targets.data * log_softmax, axis=1)
        loss = np.mean(batch_losses)
        
        return loss
    
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
        
        # Compute softmax (reuse the stability trick)
        shifted_logits = logits.data - np.max(logits.data, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Gradient is (softmax - targets) / batch_size
        batch_size = logits.shape[0]
        grad_logits = (softmax - targets.data) / batch_size
        
        # Scale by upstream gradient and return as Tensor
        return Tensor(grad_logits * grad.data), None


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
