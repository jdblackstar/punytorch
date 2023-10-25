import numpy as np

from punytorch.tensor import Tensor


def test_Add():
    x = Tensor([8])
    y = Tensor([5])
    z = x + y
    assert z.data == 13
    z.backward()
    assert np.all(x.grad == np.array([1]))


def test_Sub():
    x = Tensor([8])
    y = Tensor([5])
    z = x - y
    assert z.data == 3
    z.backward()
    assert np.all(x.grad == np.array([1]))


def test_Mul():
    x = Tensor([8])
    y = Tensor([5])
    z = x * y
    assert z.data == 40
    z.backward()
    assert x.grad == y.data
    assert y.grad == x.data


def test_TrueDiv():
    x = Tensor([8])
    y = Tensor([2])
    z = x / y
    assert z.data == 4
    z.backward()
    assert x.grad == 1 / y.data
    assert y.grad == -x.data / (y.data**2)


def test_MatMul():
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))  # 2x3 matrix
    y = Tensor(np.array([[7, 8], [9, 10], [11, 12]]))  # 3x2 matrix
    z = x @ y
    assert np.all(z.data == np.array([[58, 64], [139, 154]]))
    z.grad = np.ones_like(z.data)  # Set an initial gradient for z
    z.backward()
    assert np.allclose(x.grad, np.array([[15, 19, 23], [15, 19, 23]]))
    assert np.allclose(y.grad, np.array([[5, 5], [7, 7], [9, 9]]))
