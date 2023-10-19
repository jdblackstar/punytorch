import numpy as np


class Tensor:
    def __init__(self, data) -> None:
        self.data = data if isinstance(data, np.ndarray) else np.array(data)

    """
    BINARY OPS
    """

    def __add__(self, other) -> "Tensor":
        return Tensor(self.data + other.data)

    def __sub__(self, other) -> "Tensor":
        return Tensor(self.data - other.data)

    def __mul__(self, other) -> "Tensor":
        return Tensor(self.data * other.data)

    def __truediv__(self, other) -> "Tensor":
        return Tensor(self.data / other.data)

    def __mod__(self, other) -> "Tensor":
        return Tensor(self.data % other.data)

    def __pow__(self, other) -> "Tensor":
        return Tensor(self.data**other.data)

    def __ge__(self, other) -> "Tensor":
        return Tensor(self.data >= other.data)

    def __eq__(self, other) -> "Tensor":
        return Tensor(self.data == other.data)

    def __ne__(self, other) -> "Tensor":
        return Tensor(self.data != other.data)

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
