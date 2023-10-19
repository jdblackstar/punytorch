import numpy as np
import pytest

from punytorch import (
    Tensor
)


def test_tensor_operations():
    x = Tensor([8])
    y = Tensor([5])
    z = x + y
    assert np.allclose(z.data, np.array([13])), "x + y != 13"
