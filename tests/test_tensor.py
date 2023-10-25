import numpy as np

from punytorch.tensor import Tensor


def test_Tensor():
    x = Tensor([1, 2, 3])
    assert isinstance(x.data, np.ndarray)
    assert x.data.tolist() == [1, 2, 3]
