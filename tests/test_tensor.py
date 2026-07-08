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
    assert np.allclose(y.grad, [2.0, 3.0, 4.0])
    assert np.allclose(x.grad, [4.0, 5.0, 6.0])


def test_getitem_slice_backward_scatter():
    x = Tensor(np.arange(6.0), requires_grad=True)

    y = (x[2:5] * Tensor([1.0, 2.0, 3.0])).sum()
    y.backward()

    np.testing.assert_allclose(x.grad, [0.0, 0.0, 1.0, 2.0, 3.0, 0.0])


def test_getitem_repeated_index_backward_accumulates():
    x = Tensor(np.arange(4.0), requires_grad=True)

    y = x[[1, 1, 3]].sum()
    y.backward()

    np.testing.assert_allclose(x.grad, [0.0, 2.0, 0.0, 1.0])


def test_stack_backward_propagates_to_sources():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = Tensor([3.0, 4.0], requires_grad=True)

    z = Tensor.stack([x, y]).sum()
    z.backward()

    np.testing.assert_allclose(x.grad, [1.0, 1.0])
    np.testing.assert_allclose(y.grad, [1.0, 1.0])


def test_stack_backward_respects_axis():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = Tensor([3.0, 4.0], requires_grad=True)

    z = (Tensor.stack([x, y], axis=-1) * Tensor([[1.0, 3.0], [2.0, 4.0]])).sum()
    z.backward()

    np.testing.assert_allclose(x.grad, [1.0, 2.0])
    np.testing.assert_allclose(y.grad, [3.0, 4.0])


def test_cat_backward_propagates_to_sources():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = Tensor([3.0, 4.0], requires_grad=True)

    z = Tensor.cat([x, y]).sum()
    z.backward()

    np.testing.assert_allclose(x.grad, [1.0, 1.0])
    np.testing.assert_allclose(y.grad, [1.0, 1.0])


def test_cat_backward_respects_dim():
    x = Tensor([[1.0], [2.0]], requires_grad=True)
    y = Tensor([[3.0, 4.0], [5.0, 6.0]], requires_grad=True)

    z = (Tensor.cat([x, y], dim=-1) * Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).sum()
    z.backward()

    np.testing.assert_allclose(x.grad, [[1.0], [4.0]])
    np.testing.assert_allclose(y.grad, [[2.0, 3.0], [5.0, 6.0]])
