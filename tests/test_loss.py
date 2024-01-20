import numpy as np

from punytorch.losses import (
    BinaryCrossEntropyLoss,
    CategoricalCrossEntropyLoss,
    MSELoss,
)
from punytorch.tensor import Tensor


def test_MSELoss():
    y_true = Tensor([1.0, 2.0, 3.0])
    y_pred = Tensor([1.0, 2.0, 3.0])
    assert np.isclose(MSELoss.forward(y_pred.data, y_true.data).data, Tensor(0.0).data)
    assert np.allclose(
        MSELoss.backward(y_pred.data, y_true.data).data, Tensor([0.0, 0.0, 0.0]).data
    )


def test_BinaryCrossEntropyLoss():
    y_true = Tensor([1.0, 1.0, 1.0])
    y_pred = Tensor([0.9, 0.8, 0.9])
    assert np.isclose(
        BinaryCrossEntropyLoss.forward(y_pred.data, y_true.data).data,
        0.14462152754328741,
    )
    assert np.allclose(
        BinaryCrossEntropyLoss.backward(y_pred.data, y_true.data),
        np.array([-1.11111111, -1.25, -1.11111111]),
    )


def test_CategoricalCrossEntropyLoss():
    y_true = Tensor([1.0, 0.0, 0.0])
    y_pred = Tensor([0.7, 0.2, 0.1])
    assert np.isclose(
        CategoricalCrossEntropyLoss.forward(y_pred.data, y_true.data).data,
        0.35667494393873245,
    )
    assert np.allclose(
        CategoricalCrossEntropyLoss.backward(y_pred.data, y_true.data),
        np.array([-1.42857143, 0.0, 0.0]),
    )
