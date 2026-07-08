import numpy as np
import pytest

from punytorch.ops import Function
from punytorch.losses import (
    MSELoss,
    CrossEntropyLoss,
    BinaryCrossEntropyLoss,
    CategoricalCrossEntropyLoss,
)
from punytorch.tensor import Tensor


def loss_backward(loss_cls, y_pred, y_true, upstream=1.0):
    context = Function(loss_cls, y_pred, y_true)
    grad_pred, grad_true = loss_cls.backward(context, Tensor(upstream))
    assert grad_true is None
    return grad_pred.data


def test_MSELoss():
    y_true = Tensor([1.0, 2.0, 3.0])
    y_pred = Tensor([1.0, 3.0, 5.0])
    assert np.isclose(MSELoss.forward(y_pred, y_true), 5.0 / 3.0)
    assert np.isclose(MSELoss.forward(y_pred.data, y_true.data), 5.0 / 3.0)
    assert np.allclose(
        loss_backward(MSELoss, y_pred, y_true),
        np.array([0.0, 2.0 / 3.0, 4.0 / 3.0]),
    )


@pytest.mark.gpt
@pytest.mark.mnist
def test_CrossEntropyLoss():
    y_true_1d = Tensor([1.0, 0.0, 0.0])
    logits_1d = Tensor([0.7, 0.2, 0.1])
    y_true_2d = Tensor([[1.0, 0.0, 0.0]])
    logits_2d = Tensor([[0.7, 0.2, 0.1]])

    assert np.isclose(
        CrossEntropyLoss.forward(logits_1d, y_true_1d),
        0.7679495489036248,
    )
    assert np.allclose(
        loss_backward(CrossEntropyLoss, logits_1d, y_true_1d),
        np.array([-0.53603657, 0.28140804, 0.25462853]),
    )

    assert np.isclose(
        CrossEntropyLoss.forward(logits_2d.data, y_true_2d.data),
        0.7679495489036248,
    )
    assert np.allclose(
        loss_backward(CrossEntropyLoss, logits_2d, y_true_2d),
        np.array([[-0.53603657, 0.28140804, 0.25462853]]),
    )


def test_CrossEntropyLoss_tensor_graph():
    logits = Tensor([[0.7, 0.2, 0.1]], requires_grad=True)
    targets = Tensor([[1.0, 0.0, 0.0]])

    loss = logits.cross_entropy(targets)
    assert np.isclose(loss.data, 0.7679495489036248)

    loss.backward()
    assert np.allclose(
        logits.grad,
        np.array([[-0.53603657, 0.28140804, 0.25462853]]),
    )


def test_BinaryCrossEntropyLoss():
    y_true = Tensor([1.0, 1.0, 1.0])
    y_pred = Tensor([0.9, 0.8, 0.9])
    assert np.isclose(
        BinaryCrossEntropyLoss.forward(y_pred, y_true),
        0.14462152754328741,
    )
    assert np.allclose(
        loss_backward(BinaryCrossEntropyLoss, y_pred, y_true),
        np.array([-0.37037037, -0.41666667, -0.37037037]),
    )


def test_CategoricalCrossEntropyLoss():
    y_true = Tensor([1.0, 0.0, 0.0])
    y_pred = Tensor([0.7, 0.2, 0.1])
    assert np.isclose(
        CategoricalCrossEntropyLoss.forward(y_pred, y_true),
        0.35667494393873245,
    )
    assert np.allclose(
        loss_backward(CategoricalCrossEntropyLoss, y_pred, y_true),
        np.array([-1.42857143, 0.0, 0.0]),
    )
