import numpy as np
import pytest

from punytorch.tensor import Tensor


EPS = 1e-6
ATOL = 2e-5
RTOL = 2e-5


def _as_array(data):
    return np.array(data, dtype=np.float64, copy=True)


def _scalar_value(output, upstream):
    data = np.asarray(output.data, dtype=np.float64)
    if upstream is None:
        assert data.size == 1
        return float(data.reshape(-1)[0])
    return float(np.sum(data * upstream))


def _make_tensors(arrays, wrt):
    return [
        Tensor(array.copy(), requires_grad=index in wrt)
        for index, array in enumerate(arrays)
    ]


def _finite_difference(fn, arrays, arg_index, upstream):
    expected = np.zeros_like(arrays[arg_index], dtype=np.float64)

    for index in np.ndindex(expected.shape):
        plus = [array.copy() for array in arrays]
        minus = [array.copy() for array in arrays]
        plus[arg_index][index] += EPS
        minus[arg_index][index] -= EPS

        plus_output = fn(*_make_tensors(plus, wrt=()))
        minus_output = fn(*_make_tensors(minus, wrt=()))
        expected[index] = (_scalar_value(plus_output, upstream) - _scalar_value(minus_output, upstream)) / (
            2 * EPS
        )

    return expected


def assert_gradcheck(fn, inputs, *, upstream=None, wrt=None):
    arrays = [_as_array(data) for data in inputs]
    if wrt is None:
        wrt = tuple(range(len(arrays)))
    if upstream is not None:
        upstream = _as_array(upstream)

    tensors = _make_tensors(arrays, wrt)
    output = fn(*tensors)
    output.backward(None if upstream is None else Tensor(upstream))

    for arg_index in wrt:
        expected = _finite_difference(fn, arrays, arg_index, upstream)
        actual = tensors[arg_index].grad
        assert actual is not None
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    ("fn", "inputs", "upstream"),
    [
        (
            lambda x, y: x + y,
            ([1.0, -2.0, 0.5], [-0.5, 3.0, 2.0]),
            [0.25, -1.0, 2.0],
        ),
        (
            lambda x, y: x - y,
            ([1.0, -2.0, 0.5], [-0.5, 3.0, 2.0]),
            [0.25, -1.0, 2.0],
        ),
        (
            lambda x, y: x * y,
            ([1.0, -2.0, 0.5], [-0.5, 3.0, 2.0]),
            [0.25, -1.0, 2.0],
        ),
        (
            lambda x, y: x / y,
            ([1.0, -2.0, 0.5], [2.0, -4.0, 0.75]),
            [0.25, -1.0, 2.0],
        ),
        (
            lambda x, y: x**y,
            ([0.8, 1.2, 1.7], [1.5, 2.0, 0.5]),
            [0.25, -1.0, 2.0],
        ),
    ],
)
def test_elementwise_vector_gradcheck(fn, inputs, upstream):
    assert_gradcheck(fn, inputs, upstream=upstream)


@pytest.mark.parametrize(
    ("fn", "inputs", "upstream"),
    [
        (
            lambda x, y: x + y,
            ([1.0, -2.0, 0.5], 0.75),
            [0.25, -1.0, 2.0],
        ),
        (
            lambda x, y: x - y,
            ([1.0, -2.0, 0.5], 0.75),
            [0.25, -1.0, 2.0],
        ),
        (
            lambda x, y: x * y,
            ([1.0, -2.0, 0.5], 0.75),
            [0.25, -1.0, 2.0],
        ),
        (
            lambda x, y: x / y,
            ([1.0, -2.0, 0.5], 1.5),
            [0.25, -1.0, 2.0],
        ),
        (
            lambda x, y: x**y,
            ([0.8, 1.2, 1.7], 1.5),
            [0.25, -1.0, 2.0],
        ),
    ],
)
def test_elementwise_scalar_operand_gradcheck(fn, inputs, upstream):
    assert_gradcheck(fn, inputs, upstream=upstream)


def test_matmul_gradcheck():
    assert_gradcheck(
        lambda x, y: x @ y,
        (
            [[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]],
            [[0.5, -1.0], [2.0, 0.75], [-1.5, 1.25]],
        ),
        upstream=[[0.5, -1.0], [1.5, 0.25]],
    )


@pytest.mark.parametrize(
    ("fn", "inputs"),
    [
        (lambda x: x.sum(), ([[1.0, -2.0, 0.5], [3.0, -1.5, 2.0]],)),
        (lambda x: x.mean(), ([[1.0, -2.0, 0.5], [3.0, -1.5, 2.0]],)),
    ],
)
def test_reduction_scalar_gradcheck(fn, inputs):
    assert_gradcheck(fn, inputs)


@pytest.mark.parametrize(
    ("fn", "inputs", "upstream"),
    [
        (
            lambda x: x.sum(axis=1),
            ([[1.0, -2.0, 0.5], [3.0, -1.5, 2.0]],),
            [0.5, -1.5],
        ),
        (
            lambda x: x.mean(axis=0),
            ([[1.0, -2.0, 0.5], [3.0, -1.5, 2.0]],),
            [0.5, -1.5, 2.0],
        ),
    ],
)
def test_reduction_vector_gradcheck(fn, inputs, upstream):
    assert_gradcheck(fn, inputs, upstream=upstream)


def test_reshape_gradcheck():
    assert_gradcheck(
        lambda x: x.reshape(3, 2),
        ([[1.0, -2.0, 0.5], [3.0, -1.5, 2.0]],),
        upstream=[[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0]],
    )


@pytest.mark.parametrize(
    ("fn", "inputs", "upstream"),
    [
        (lambda x: x.relu(), ([-1.5, -0.2, 0.4, 2.0],), [0.5, -1.0, 1.5, 0.25]),
        (lambda x: x.sigmoid(), ([-1.5, -0.2, 0.4, 2.0],), [0.5, -1.0, 1.5, 0.25]),
    ],
)
def test_activation_gradcheck(fn, inputs, upstream):
    assert_gradcheck(fn, inputs, upstream=upstream)


def test_cross_entropy_2d_logits_gradcheck():
    assert_gradcheck(
        lambda logits, targets: logits.cross_entropy(targets),
        (
            [[1.2, -0.7, 0.3], [-0.4, 0.9, 1.5]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        wrt=(0,),
    )
