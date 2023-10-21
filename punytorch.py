import numpy as np


class Tensor:
    def __init__(self, data) -> None:
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.context = None

    """
    BINARY OPS
    """

    def __add__(self, other) -> "Tensor":
        fn = Function(Add, self, other)
        result = Add.forward(self, other)
        result.context = fn
        return result

    def __sub__(self, other) -> "Tensor":
        fn = Function(Sub, self, other)
        result = Sub.forward(self, other)
        result.context = fn
        return result

    def __mul__(self, other) -> "Tensor":
        fn = Function(Mul, self, other)
        result = Mul.forward(self, other)
        result.context = fn
        return result

    def __truediv__(self, other) -> "Tensor":
        return Tensor(self.data / other.data)

    def __mod__(self, other) -> "Tensor":
        return Tensor(self.data % other.data)

    def __pow__(self, other) -> "Tensor":
        return Tensor(self.data**other.data)

    """
    UNARY OPS
    """

    def __abs__(self) -> "Tensor":
        return Tensor(np.abs(self.data))

    def __neg__(self) -> "Tensor":
        return Tensor(-self.data)

    def __invert__(self) -> "Tensor":
        return Tensor(~self.data)

    def __repr__(self) -> str:
        return f"tensor({self.data})"


class Function:
    def __init__(self, op, *args):
        self.op = op
        self.args = args


class Add:
    @staticmethod
    def forward(x, y):
        """
        z = x + y
        """
        return Tensor(x.data + y.data)

    @staticmethod
    def backward(context, grad):
        """
        d(x + y)/dx = 1
        d(x + y)/dy = 1
        so return Tensor([1]) for both x and y
        """
        x, y = context.args
        return Tensor([1]), Tensor([1])


class Sub:
    @staticmethod
    def forward(x, y):
        """
        z = x - y
        """
        return Tensor(x.data - y.data)

    @staticmethod
    def backward(context, grad):
        """
        d(x - y)/dx = 1
        d(x - y)/dy = -
        so return Tensor([1]) for x and Tensor([-1]) for y
        """
        x, y = context.args
        return Tensor([1]), Tensor([-1])


class Mul:
    @staticmethod
    def forward(x, y):
        """
        z = x * y
        """
        return Tensor(x.data * y.data)

    @staticmethod
    def backward(context, grad):
        """
        d(x * y)/dx = y
        d(x * y)/dy = x
        so return Tensor(y.data) for x and Tensor(x.data) for y
        """
        x, y = context.args
        return Tensor(y.data), Tensor(x.data)


class TrueDiv:
    @staticmethod
    def forward(x, y):
        """
        z = x / y
        """
        return Tensor(x.data / y.data)

    @staticmethod
    def backward(context, grad):
        """
        d(x / y)/dx = 1/y
        d(x / y)/dy = -x/y^2
        """
        x, y = context.args
        return Tensor(grad.data / y.data), Tensor(-grad.data * x.data / (y.data**2))


class Mod:
    """
    WARNING: The modulus operation is not differentiable at integer points,
    and the derivative with respect to `y` is undefined. This implementation
    assumes that the gradient with respect to `y` is 0, which is not universally
    accepted. It also checks if all elements in `y.data` are integers and raises
    a ValueError if they're not. This is a simplification and may not be suitable
    for all use cases.
    """

    @staticmethod
    def forward(x, y):
        """
        z = x % y
        """
        return Tensor(x.data % y.data)

    @staticmethod
    def backward(context, grad):
        """
        d(x % y)/dx = 1
        d(x % y)/dy = 0
        """
        x, y = context.args
        if not np.all(y.data.astype(int) == y.data):
            raise ValueError(
                "Backward operation for modulus is only defined when y is an integer."
            )
        return Tensor(np.ones_like(x.data)), Tensor(np.zeros_like(y.data))


class Pow:
    @staticmethod
    def forward(x, y):
        """
        z = x ^ y
        """
        return Tensor(x.data**y.data)

    @staticmethod
    def backward(context, grad):
        """
        d(x ^ y)/dx = y * x^(y - 1)
        d(x ^ y)/dy = x^y * log(x)
        """
        x, y = context.args
        return Tensor(y.data * x.data ** (y.data - 1)), Tensor(
            x.data**y.data * np.log(x.data)
        )
