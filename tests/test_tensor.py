import numpy as np
import pytest

from punytorch.tensor import Tensor


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
    """
    The gradient of z with respect to y is 1 (from the + y part of the operation),
    plus the value of x (from the x * y part of the operation),
    so y.grad = [2.0, 3.0, 4.0]
    
    The gradient of z with respect to x is the value of y (from the x * y part of the operation),
    so x.grad = [4.0, 5.0, 6.0]
    """
    assert np.allclose(y.grad.data, [2.0, 3.0, 4.0])
    assert np.allclose(x.grad.data, [4.0, 5.0, 6.0])
