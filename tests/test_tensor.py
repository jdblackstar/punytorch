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


def test_no_grad_operations_do_not_track_gradients():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    with Tensor.no_grad():
        y = x + 1
        z = y.sum()
        relu = x.relu()

    for result in (y, z, relu):
        assert result.requires_grad is False
        assert result.context is None


def test_no_grad_decorator_disables_gradient_tracking():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    @Tensor.no_grad()
    def add_and_sum(tensor):
        return (tensor + 1).sum()

    result = add_and_sum(x)

    assert result.requires_grad is False
    assert result.context is None


def test_no_grad_nested_context_restores_outer_state():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    with Tensor.no_grad():
        with Tensor.no_grad():
            inner = x + 1
        outer = x * 2

    normal = x + 3

    assert inner.requires_grad is False
    assert inner.context is None
    assert outer.requires_grad is False
    assert outer.context is None
    assert normal.requires_grad is True
    assert normal.context is not None


def test_no_grad_does_not_disable_later_backpropagation():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    with Tensor.no_grad():
        ignored = (x * 10).sum()

    z = (x * 2).sum()
    z.backward()

    assert ignored.requires_grad is False
    assert ignored.context is None
    assert z.requires_grad is True
    assert z.context is not None
    assert np.allclose(x.grad.data, [2.0, 2.0, 2.0])
