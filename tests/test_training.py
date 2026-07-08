import numpy as np

from punytorch.nn.modules import Linear, Module
from punytorch.nn.optimizers import SGD
from punytorch.tensor import Tensor


class TinyClassifier(Module):
    def __init__(self):
        self.linear = Linear(2, 2)
        self.linear.weight.data = np.array([[0.1, -0.2], [-0.1, 0.2]], dtype=np.float64)
        self.linear.bias.data = np.zeros(2, dtype=np.float64)

    def forward(self, x):
        return self.linear(x)


def test_tiny_classifier_learns_linearly_separable_data():
    inputs = Tensor(
        np.array(
            [
                [-2.0, -1.0],
                [-1.5, -2.0],
                [1.0, 2.0],
                [2.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    targets = Tensor(
        np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    model = TinyClassifier()
    optimizer = SGD(model.parameters(), lr=0.1)

    initial_loss = model(inputs).cross_entropy(targets).item()

    for _ in range(80):
        optimizer.zero_grad()
        loss = model(inputs).cross_entropy(targets)
        loss.backward()
        optimizer.step()

    logits = model(inputs)
    final_loss = logits.cross_entropy(targets).item()
    predictions = np.argmax(logits.data, axis=1)

    assert final_loss < initial_loss * 0.2
    assert predictions.tolist() == [0, 0, 1, 1]
