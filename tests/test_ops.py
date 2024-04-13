from __future__ import annotations
import logging

import numpy as np
import pytest

from punytorch.tensor import Tensor

logger = logging.getLogger()


def test_Add():
    """
    Tests the Add operation located in punytorch.ops, implemented in punytorch.tensor.Tensor.__add__
    """
    # forward test
    x = Tensor([8], requires_grad=True)
    y = Tensor([5], requires_grad=True)
    z = x + y
    assert z.context is not None, "z.context should not be None. Check that requires_grad=True."
    assert np.all(z.data == 13)

    # backward test
    z.backward(Tensor(np.ones_like(z.data)))
    assert np.all(x.grad.data == np.array([1]))
    assert np.all(y.grad.data == np.array([1]))


def test_Sub():
    """
    Tests the Sub operation located in punytorch.ops, implemented in punytorch.tensor.Tensor.__sub__
    """
    # forward test
    x = Tensor([8], requires_grad=True)
    y = Tensor([5], requires_grad=True)
    z = x - y
    assert z.context is not None, "z.context should not be None. Check that requires_grad=True."
    assert np.all(z.data == 3)

    # backward test
    z.backward(Tensor(np.ones_like(z.data)))
    assert np.all(x.grad.data == np.array([1]))
    assert np.all(y.grad.data == np.array([-1]))


def test_Mul():
    """
    Tests the Mul operation located in punytorch.ops, implemented in punytorch.tensor.Tensor.__mul__
    """
    # forward test
    x = Tensor([8], requires_grad=True)
    y = Tensor([5], requires_grad=True)
    z = x * y  # Use the * operator
    assert z.context is not None, "z.context should not be None. Check that requires_grad=True."
    assert np.all(z.data == 40)

    # backward test
    z.backward(Tensor(np.ones_like(z.data)))
    assert np.all(x.grad.data == np.array([5]))
    assert np.all(y.grad.data == np.array([8]))


def test_TrueDiv():
    """
    Tests the TrueDiv operation located in punytorch.ops, implemented in punytorch.tensor.Tensor.__truediv__
    """
    # forward test
    x = Tensor([8], requires_grad=True)
    y = Tensor([2], requires_grad=True)
    z = x / y  # Use the / operator
    assert z.context is not None, "z.context should not be None. Check that requires_grad=True."
    assert np.all(z.data == 4)

    # backward test
    z.backward(Tensor(np.ones_like(z.data)))
    assert np.allclose(x.grad.data, np.array([1 / 2]))
    assert np.allclose(y.grad.data, np.array([-8 / 4]))


def test_MatMul():
    """
    Tests the MatMul operation located in punytorch.ops, implemented in punytorch.tensor.Tensor.__matmul__
    """
    # forward test
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), requires_grad=True)  # 2x3 matrix
    y = Tensor(np.array([[7, 8], [9, 10], [11, 12]]), requires_grad=True)  # 3x2 matrix
    z = x @ y
    assert z.context is not None, "z.context should not be None. Check that requires_grad=True."
    assert np.all(z.data == np.array([[58, 64], [139, 154]]))

    # backward test
    z.backward(Tensor(np.ones_like(z.data)))
    assert np.allclose(x.grad.data, np.array([[15, 19, 23], [15, 19, 23]]))
    assert np.allclose(y.grad.data, np.array([[5, 5], [7, 7], [9, 9]]))
