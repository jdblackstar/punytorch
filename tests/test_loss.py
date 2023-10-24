import numpy as np

from punytorch.losses import BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss, MSELoss
from punytorch.tensor import Tensor


def test_MSELoss():
    y_true = Tensor(np.array([1.0, 2.0, 3.0]))
    y_pred = Tensor(np.array([1.0, 2.0, 3.0]))
    assert MSELoss.forward(y_pred.data, y_true.data) == 0.0
    assert np.all(MSELoss.backward(y_pred.data, y_true.data) == 0.0)


def test_BinaryCrossEntropyLoss():
    y_true = Tensor(np.array([1.0, 0.0, 1.0]))
    y_pred = Tensor(np.array([0.9, 0.1, 0.9]))
    assert (
        BinaryCrossEntropyLoss.forward(y_pred.data, y_true.data) == 0.10536051565782628
    )
    assert np.all(
        BinaryCrossEntropyLoss.backward(y_pred.data, y_true.data)
        == np.array([0.55555556, -1.11111111, 0.55555556])
    )


def test_CategoricalCrossEntropyLoss():
    y_true = Tensor(np.array([1.0, 0.0, 0.0]))
    y_pred = Tensor(np.array([0.7, 0.2, 0.1]))
    assert (
        CategoricalCrossEntropyLoss.forward(y_pred.data, y_true.data)
        == 0.35667494393873245
    )
    assert np.all(
        CategoricalCrossEntropyLoss.backward(y_pred.data, y_true.data)
        == np.array([-1.42857143, 0.0, 0.0])
    )
