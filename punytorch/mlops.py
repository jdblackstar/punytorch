from punytorch.ops import Function


class Reshape(Function):
    @staticmethod
    def forward(x, shape):
        return x.__class__(x.data.reshape(tuple(shape)))

    @staticmethod
    def backward(ctx: Function, grad):
        x, _ = ctx.args
        return grad.__class__(grad.data.reshape(x.shape)), None
