import numpy as np

from punytorch.tensor import Tensor


def test_Tensor():
    x = Tensor([1, 2, 3])
    assert isinstance(x.data, np.ndarray)
    assert x.data.tolist() == [1, 2, 3]


def test_backpropagation():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    z = x * y + y
    assert z.data.tolist() == [8.0, 15.0, 24.0]
    z.backward()
    assert x.grad.tolist() == [4.0, 5.0, 6.0]
    assert y.grad.tolist() == [2.0, 3.0, 4.0]
