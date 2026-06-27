import numpy as np
import pytest

from punytorch.nn.modules import Linear, Module, ModuleList, Parameter
from punytorch.nn.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSProp
from punytorch.tensor import Tensor


class NestedModel(Module):
    def __init__(self):
        self.first = Linear(2, 3)
        self.layers = ModuleList([Linear(3, 2), Linear(2, 1, bias=False)])
        self.scale = Parameter(np.array([1.0]))


class SharedParameterModel(NestedModel):
    def __init__(self):
        super().__init__()
        self.alias = self.first.weight


def test_parameters_are_in_deterministic_traversal_order_without_duplicates():
    model = SharedParameterModel()

    assert model.parameters() == [
        model.first.weight,
        model.first.bias,
        model.layers[0].weight,
        model.layers[0].bias,
        model.layers[1].weight,
        model.scale,
    ]


def test_state_dict_round_trips_nested_module_list_dotted_keys():
    source = NestedModel()
    source.first.weight.data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    source.first.bias.data = np.array([7.0, 8.0, 9.0])
    source.layers[0].weight.data = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
    source.layers[0].bias.data = np.array([16.0, 17.0])
    source.layers[1].weight.data = np.array([[18.0, 19.0]])
    source.scale.data = np.array([20.0])

    state = source.state_dict()

    assert list(state.keys()) == [
        "first.weight",
        "first.bias",
        "layers.0.weight",
        "layers.0.bias",
        "layers.1.weight",
        "scale",
    ]

    target = NestedModel()
    target.load_state_dict(state)

    for key, value in state.items():
        np.testing.assert_allclose(target.state_dict()[key].data, value.data)

    loaded = target.first.weight.data.copy()
    state["first.weight"].data.fill(-1.0)
    np.testing.assert_allclose(target.first.weight.data, loaded)


def test_linear_without_bias_has_no_bias_parameter_and_forward_works():
    layer = Linear(2, 1, bias=False)
    layer.weight.data = np.array([[2.0, 3.0]])

    output = layer(Tensor(np.array([[4.0, 5.0]])))

    assert layer.bias is None
    assert layer.parameters() == [layer.weight]
    assert list(layer.state_dict().keys()) == ["weight"]
    np.testing.assert_allclose(output.data, np.array([[23.0]]))


@pytest.mark.parametrize(
    "optimizer_factory",
    [
        lambda params: SGD(params, lr=0.1),
        lambda params: Adam(params, lr=0.1),
        lambda params: RMSProp(params, lr=0.1),
        lambda params: Adagrad(params, lr=0.1),
        lambda params: Adadelta(params),
        lambda params: Adamax(params, lr=0.1),
    ],
)
@pytest.mark.parametrize(
    "grad",
    [
        np.array([0.25, -0.5]),
        Tensor(np.array([0.25, -0.5])),
    ],
)
def test_optimizers_accept_array_and_tensor_grads(optimizer_factory, grad):
    param = Parameter(np.array([1.0, -2.0]))
    param.grad = grad
    optimizer = optimizer_factory([param])

    before = param.data.copy()
    optimizer.step()

    assert np.all(np.isfinite(param.data))
    assert not np.allclose(param.data, before)


def test_zero_grad_handles_tensor_grad_and_materializes_array():
    param = Parameter(np.array([1.0, -2.0]))
    param.grad = Tensor(np.array([0.25, -0.5]))
    optimizer = SGD((item for item in [param]), lr=0.1)

    optimizer.zero_grad()

    assert isinstance(param.grad, np.ndarray)
    np.testing.assert_allclose(param.grad, np.zeros(2))
