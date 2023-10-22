import numpy as np

from tensor import Tensor

if __name__ == "__main__":
    x = Tensor([8])
    y = Tensor([5])
    z = x * y
    y = Tensor([5])

    print("Add")
    z = x + y
    print(z)
    z.backward()
    print(f"x: {x} , grad {x.grad}")
    print(f"y: {y} , grad {y.grad}")
    print("=" * 100)

    print("Mul")
    z = x * y
    print(z)

    z.backward()
    print(f"x: {x} , grad {x.grad}")
    print(f"y: {y} , grad {y.grad}")
    print("=" * 100)
