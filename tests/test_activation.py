import numpy as np
import pytest

from punytorch.activations import ReLU, Sigmoid, Softmax
from punytorch.tensor import Tensor


@pytest.mark.gpt
@pytest.mark.mnist
def test_ReLU():
    x = Tensor(np.array([-2, -1, 0, 1, 2]), requires_grad=True)
    y = x.relu()
    assert np.all(y.data == np.array([0, 0, 0, 1, 2]))
    y.backward(np.ones_like(x.data))
    assert np.all(x.grad == np.array([0, 0, 0, 1, 1]))


@pytest.mark.gpt
def test_Sigmoid():
    x = Tensor(np.array([0, 1, 2, 3, 4, 5]), requires_grad=True)
    y = x.sigmoid()
    assert np.allclose(y.data, 1 / (1 + np.exp(-x.data)))
    y.backward()
    assert np.allclose(x.grad, y.data * (1 - y.data))


def test_activation_forwards_accept_ndarrays():
    x = np.array([-1.0, 0.0, 1.0])

    assert np.allclose(ReLU.forward(x), np.array([0.0, 0.0, 1.0]))
    assert np.allclose(Sigmoid.forward(x), 1 / (1 + np.exp(-x)))
    assert np.allclose(Softmax.forward(x).sum(), 1.0)


def test_sigmoid_backward_with_upstream_gradient():
    x = Tensor(np.array([0.0, 1.0, 2.0]), requires_grad=True)
    y = x.sigmoid()
    upstream = np.array([1.0, 2.0, 3.0])

    y.backward(upstream)

    assert np.allclose(x.grad, upstream * y.data * (1 - y.data))


@pytest.mark.gpt
def test_Softmax():
    x = Tensor(np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]), requires_grad=True)
    y = x.softmax()

    expected = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
    expected /= np.sum(expected, axis=-1, keepdims=True)
    assert np.allclose(y.data, expected)
    assert y.data.shape == x.data.shape
    assert np.allclose(y.data.sum(axis=-1), np.ones(x.data.shape[0]))

    upstream = np.array([[1.0, 0.0, -1.0], [0.5, -0.5, 1.0]])
    y.backward(upstream)
    expected_grad = expected * (
        upstream - np.sum(upstream * expected, axis=-1, keepdims=True)
    )
    assert x.grad.shape == x.data.shape
    assert np.allclose(x.grad, expected_grad)
