import logging

import numpy as np
import pytest

from punytorch.losses import (
    MSELoss,
    CrossEntropyLoss,
    BinaryCrossEntropyLoss,
    CategoricalCrossEntropyLoss,
)
from punytorch.tensor import Tensor

logger = logging.getLogger()


def test_MSELoss():
    y_true = Tensor([1.0, 2.0, 3.0])
    y_pred = Tensor([1.0, 2.0, 3.0])
    assert np.isclose(MSELoss.forward(y_pred.data, y_true.data).data, Tensor(0.0).data)
    assert np.allclose(MSELoss.backward(y_pred.data, y_true.data).data, Tensor([0.0, 0.0, 0.0]).data)


@pytest.mark.gpt
@pytest.mark.mnist
def test_CrossEntropyLoss():
    # create and test both 1-dimensional and 2-dimensional tensors
    y_true_1d = Tensor([1.0, 0.0, 0.0])
    y_pred_1d = Tensor([0.7, 0.2, 0.1])
    y_true_2d = Tensor([[1.0, 0.0, 0.0]])
    y_pred_2d = Tensor([[0.7, 0.2, 0.1]])

    # test of 1-dimensional tensor
    assert np.isclose(
        CrossEntropyLoss.forward(y_pred_1d.data, y_true_1d.data).data,
        0.2559831829678749,
    )
    assert np.allclose(
        CrossEntropyLoss.backward(y_pred_1d.data, y_true_1d.data).data,
        np.array([-0.17867886, 0.09380268, 0.08487618]),
    )

    # test of 2-dimensional tensor
    assert np.isclose(
        CrossEntropyLoss.forward(y_pred_2d.data, y_true_2d.data).data,
        0.7679495489036248,
    )
    assert np.allclose(
        CrossEntropyLoss.backward(y_pred_2d.data, y_true_2d.data).data,
        np.array([[-0.53603657, 0.28140804, 0.25462853]]),
    )


def test_BinaryCrossEntropyLoss():
    y_true = Tensor([1.0, 1.0, 1.0])
    y_pred = Tensor([0.9, 0.8, 0.9])
    assert np.isclose(
        BinaryCrossEntropyLoss.forward(y_pred.data, y_true.data).data,
        0.14462152754328741,
    )
    assert np.allclose(
        BinaryCrossEntropyLoss.backward(y_pred.data, y_true.data).data,
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
        CategoricalCrossEntropyLoss.backward(y_pred.data, y_true.data).data,
        np.array([-1.42857143, 0.0, 0.0]),
    )
