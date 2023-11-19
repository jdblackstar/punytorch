from punytorch.ops import Function
from punytorch.tensor import Tensor


class Reshape(Function):
    @staticmethod
    def forward(x, shape) -> Tensor:
        return Tensor(x.data.reshape(tuple(shape)))

    @staticmethod
    def backward(ctx: Function, grad: Tensor) -> Tensor:
        x, _ = ctx.args
        return Tensor(grad.data.reshape(x.shape)), None
