import numpy as np
import pytest

from punytorch.losses import (
    MSELoss,
    CrossEntropyLoss,
    BinaryCrossEntropyLoss,
    CategoricalCrossEntropyLoss,
)
from punytorch.ops import Function
from punytorch.tensor import Tensor


def test_MSELoss():
    y_true = Tensor([1.0, 2.0, 3.0])
    y_pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    assert np.isclose(MSELoss.forward(y_pred.data, y_true.data), 0.0)

    grad_pred, grad_true = MSELoss.backward(Function(MSELoss, y_pred, y_true), 1.0)
    assert np.allclose(grad_pred, np.array([0.0, 0.0, 0.0]))
    assert grad_true is None


@pytest.mark.gpt
@pytest.mark.mnist
def test_CrossEntropyLoss():
    targets = Tensor([[1.0, 0.0, 0.0]])
    logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)

    loss = logits.cross_entropy(targets)
    assert np.isclose(loss.data, 0.41703001627783354)

    loss.backward()
    assert np.allclose(
        logits.grad,
        np.array([[-0.34099886, 0.24243297, 0.09856589]]),
    )


def test_CrossEntropyLoss_supports_1d_logits():
    targets = Tensor([1.0, 0.0, 0.0])
    logits = Tensor([2.0, 1.0, 0.1], requires_grad=True)

    loss = logits.cross_entropy(targets)
    assert np.isclose(loss.data, 0.41703001627783354)

    loss.backward()
    assert np.allclose(
        logits.grad,
        np.array([-0.34099886, 0.24243297, 0.09856589]),
    )


def test_BinaryCrossEntropyLoss():
    y_true = Tensor([1.0, 1.0, 1.0])
    y_pred = Tensor([0.9, 0.8, 0.9], requires_grad=True)
    assert np.isclose(
        BinaryCrossEntropyLoss.forward(y_pred.data, y_true.data),
        0.14462152754328741,
    )

    grad_pred, grad_true = BinaryCrossEntropyLoss.backward(
        Function(BinaryCrossEntropyLoss, y_pred, y_true),
        1.0,
    )
    assert np.allclose(grad_pred, np.array([-0.37037037, -0.41666667, -0.37037037]))
    assert grad_true is None


def test_CategoricalCrossEntropyLoss():
    y_true = Tensor([1.0, 0.0, 0.0])
    y_pred = Tensor([0.7, 0.2, 0.1], requires_grad=True)
    assert np.isclose(
        CategoricalCrossEntropyLoss.forward(y_pred.data, y_true.data),
        0.35667494393873245,
    )

    grad_pred, grad_true = CategoricalCrossEntropyLoss.backward(
        Function(CategoricalCrossEntropyLoss, y_pred, y_true),
        1.0,
    )
    assert np.allclose(grad_pred, np.array([-1.42857143, 0.0, 0.0]))
    assert grad_true is None
