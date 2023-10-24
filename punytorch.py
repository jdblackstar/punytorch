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

    print("MatMul")
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))  # 2x3 matrix
    y = Tensor(np.array([[7, 8], [9, 10], [11, 12]]))  # 3x2 matrix
    print("x:")
    print(x)
    print("y:")
    print(y)
    z = x @ y  # use @ operator for matrix multiplication
    print("z = x @ y:")
    print(z)
    z.backward()
    print("After backward:")
    print(f"x: {x} , grad {x.grad}")
    print(f"y: {y} , grad {y.grad}")
    print("=" * 100)

    print("ReLU")
    x = Tensor([-2, -1, 0, 1, 2])
    z = x.relu()
    print(x)
    print(z)
    z.backward()
    print(f"x: {x} , grad {x.grad}")
    print("=" * 100)

    print("Sigmoid")
    x = Tensor([0, 1, 2, 3, 4, 5])
    z = x.sigmoid()
    print(x)
    print(z)
    z.backward()
    print(f"x: {x} , grad {x.grad}")
    print("=" * 100)

    print("Softmax")
    x = Tensor([0, 1, 2, 3, 4, 5])
    z = x.softmax()
    print(x)
    print(z)
    z.backward()
    print(f"x: {x} , grad {x.grad}")
    print("=" * 100)
