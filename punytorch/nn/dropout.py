import numpy as np
import punytorch.nn as nn
from punytorch.tensor import Tensor


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5, seed: int = None):
        """
        Initializes the Dropout layer.

        Args:
            p (float or Tensor): The probability of an element to be zeroed. Default: 0.5
                Can be a float for a constant dropout rate, or a Tensor for element-wise rates.
            seed (int, optional): The seed for the random number generator. If provided,
                ensures reproducibility of dropout mask across runs. Default: None

        Raises:
            TypeError: If `p` is not a float or a Tensor.
        """
        super().__init__()
        self.p = p
        self.seed = seed
        if isinstance(p, float):
            self.p = float(p)
        elif isinstance(p, Tensor):
            self.p = p.data
        else:
            raise TypeError(f"p must be a float or a Tensor, got {type(p)}")

    def forward(self, input: Tensor, train: bool = True) -> Tensor:
        """
        Applies Dropout to the input Tensor during training.

        Args:
            input (Tensor): Input tensor.
            train (bool): If True, apply dropout. If False, return the input as is.

        Returns:
            Tensor: Output tensor after applying dropout.
        """
        if train:
            # Generate a mask with the same shape as the input
            # Elements are drawn from a Bernoulli distribution
            self.mask = (np.random.rand(*input.shape) > self.p) / (1 - self.p)
            return input * self.mask
        else:
            return input

    def __call__(self, input: Tensor, train: bool = True) -> Tensor:
        return self.forward(input, train)
