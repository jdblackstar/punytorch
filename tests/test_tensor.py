import logging

import numpy as np
import pytest

from punytorch.tensor import Tensor

logger = logging.getLogger()


@pytest.mark.gpt
@pytest.mark.mnist
def test_Tensor():
    x = Tensor([1, 2, 3])
    assert isinstance(x.data, np.ndarray)
    assert x.data.tolist() == [1, 2, 3]


@pytest.mark.gpt
@pytest.mark.mnist
def test_backpropagation():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    z = x * y + y
    assert np.allclose(z.data, [8.0, 15.0, 24.0])

    z.backward()  # Start the backward pass with an initial gradient of 1 for z
    logger.debug(f"z is: ", z)
    """
    The gradient of z with respect to y is 1 (from the + y part of the operation),
    plus the value of x (from the x * y part of the operation),
    so y.grad = [2.0, 3.0, 4.0]
    
    The gradient of z with respect to x is the value of y (from the x * y part of the operation),
    so x.grad = [4.0, 5.0, 6.0]
    """
    logger.debug(f"x.grad is: ", x.grad)
    logger.debug(f"y.grad is: ", y.grad)

    assert np.allclose(y.grad.data, [2.0, 3.0, 4.0]), f"y.grad.data should be [2.0, 3.0, 4.0], got {y.grad.data}"
    assert np.allclose(x.grad.data, [4.0, 5.0, 6.0]), f"x.grad.data should be [4.0, 5.0, 6.0], got {x.grad.data}"


@pytest.mark.gpt
def test_sum_on_Tensor():
    x = Tensor(np.random.rand(2, 3, 4))
    print(f"x shape: {x.shape}, x value: {x}")
    print(f"the data: {x.data}")
    summed_x = x.sum(axis=-1)
    # Assert that the shape of summed_x matches the expected shape after summing along the last dimension
    assert summed_x.shape == x.shape[:-1], f"Expected shape {x.shape[:-1]}, but got {summed_x.shape}"
