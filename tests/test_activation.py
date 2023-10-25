import numpy as np

from punytorch.tensor import Tensor


def test_ReLU():
    x = Tensor(np.array([-2, -1, 0, 1, 2]))
    y = x.relu()
    assert np.all(y.data == np.array([0, 0, 0, 1, 2]))
    y.backward(np.ones_like(x.data))
    assert np.all(x.grad == np.array([0, 0, 0, 1, 1]))


def test_Sigmoid():
    x = Tensor(np.array([0, 1, 2, 3, 4, 5]))
    y = x.sigmoid()
    assert np.allclose(y.data, 1 / (1 + np.exp(-x.data)))
    y.backward()
    assert np.allclose(x.grad, y.data * (1 - y.data))


# def test_Softmax():
#     x = Tensor(np.array([0, 1, 2, 3, 4, 5]))
#     y = x.softmax()
#     assert np.allclose(y.data, np.exp(x.data) / np.sum(np.exp(x.data)))
#     y.backward(np.ones_like(x.data))
#     # The gradient of softmax is complex and depends on the input values.
#     # Here, we're just checking that the gradient has the same shape as the input.
#     assert x.grad.shape == x.data.shape
