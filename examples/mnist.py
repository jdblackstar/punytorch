# Standard library imports
import random
import time

# Third-party imports
import numpy as np  # will remove this ASAP
from tqdm import tqdm

from datasets.mnist.fetch_mnist import download_mnist, load_mnist
from punytorch.helpers import is_one_hot
from punytorch.nn.modules import Linear, Module
from punytorch.nn.optimizers import Adam
from punytorch.tensor import Tensor

# Constants
EPOCHS = 10
BATCH_SIZE = 32
LR = 4e-3
MNIST_DIR = "datasets/mnist"


def get_batch(images: Tensor, labels: Tensor):
    indices = list(range(0, len(images), BATCH_SIZE))
    random.shuffle(indices)
    for i in indices:
        yield images[i : i + BATCH_SIZE], labels[i : i + BATCH_SIZE]


class Network(Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = Linear(28 * 28, 128)
        self.l2 = Linear(128, 64)
        self.l3 = Linear(64, 10)

    def forward(self, x: Tensor) -> Tensor:
        """
        Executes the network's forward pass and returns probabilities suitable for cross-entropy.

        Args:
            x (Tensor): The input Tensor, shape (batch_size, 28*28).

        Returns:
            Tensor: The output probabilities (batch_size, 10).
        """
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        return self.l3(x)  # Return raw logits


@Tensor.no_grad()
def test(model: Network, test_images: Tensor, test_labels: Tensor):
    preds = model.forward(test_images)
    pred_indices = np.argmax(preds.data, axis=-1)

    # Convert one-hot encoded labels to class indices
    test_labels = np.argmax(Tensor.data_to_numpy(test_labels.data), axis=-1)

    correct = 0
    for p, t in zip(pred_indices.reshape(-1), test_labels.reshape(-1)):
        if p == t:
            correct += 1
    accuracy = correct / len(test_labels)
    print(f"Test accuracy: {accuracy:.2%}")


def cross_entropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return y_pred.cross_entropy(y_true)


def train(
    model: Network,
    optimizer: Adam,
    train_images: Tensor,
    train_labels: Tensor,
):
    model.train()
    for epoch in range(EPOCHS):
        batch_generator = get_batch(train_images, train_labels)
        num_batches = len(train_images) // BATCH_SIZE
        with tqdm(total=num_batches) as pbar:
            for batch_images, batch_labels in batch_generator:
                # Convert one-hot encoded labels to class indices
                # batch_labels = np.argmax(batch_labels, axis=-1)

                optimizer.zero_grad()
                pred = model.forward(batch_images)
                loss = cross_entropy(pred, batch_labels)
                loss.backward()

                # Debug gradients after backward pass (only for first batch of first epoch)
                if epoch == 0 and pbar.n == 0:
                    _debug_grad_flow(model)

                optimizer.step()

                pbar.update(1)
                pbar.set_postfix({"loss": float(loss.item())})

        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
        test(model, test_images, test_labels)


def _debug_grad_flow(model: Module) -> None:
    """
    Prints a summary of gradients flowing through the model parameters.
    Useful for debugging backprop.
    """
    print("=== Gradient Flow Debug ===")
    for name, layer in model.__dict__.items():
        if isinstance(layer, Linear):
            print(f"Layer: {name}")
            # Check weight gradients
            if hasattr(layer.weight, "grad"):
                if layer.weight.grad is not None:
                    grad_mean = np.mean(layer.weight.grad)
                    grad_std = np.std(layer.weight.grad)
                    grad_max = np.max(np.abs(layer.weight.grad))
                    print(
                        f"  weight grad mean: {grad_mean:.6f} | std: {grad_std:.6f} | max_abs: {grad_max:.6f}"
                    )
                    print(f"  weight grad shape: {layer.weight.grad.shape}")
                else:
                    print("  weight grad: None")
            else:
                print("  weight has no grad attribute")

            # Check bias gradients
            if layer.bias is not None:
                if hasattr(layer.bias, "grad"):
                    if layer.bias.grad is not None:
                        grad_mean = np.mean(layer.bias.grad)
                        grad_std = np.std(layer.bias.grad)
                        grad_max = np.max(np.abs(layer.bias.grad))
                        print(
                            f"  bias grad mean: {grad_mean:.6f} | std: {grad_std:.6f} | max_abs: {grad_max:.6f}"
                        )
                        print(f"  bias grad shape: {layer.bias.grad.shape}")
                    else:
                        print("  bias grad: None")
                else:
                    print("  bias has no grad attribute")
    print("==========================")


if __name__ == "__main__":
    download_mnist(MNIST_DIR)
    (train_images, train_labels), (test_images, test_labels) = load_mnist(MNIST_DIR)
    assert train_labels.min() >= 0 and train_labels.max() < 10, "Invalid labels"
    assert train_images.min() >= 0 and train_images.max() <= 255, "Invalid images"

    # using numpy for now, but we'll work on a custom implentation later maybe
    num_classes = 10
    train_labels = np.eye(num_classes)[train_labels]
    test_labels = np.eye(num_classes)[test_labels]

    # Check if the labels are one-hot encoded
    print(f"Train labels one-hot encoded: {is_one_hot(train_labels)}")
    print(f"Test labels one-hot encoded: {is_one_hot(test_labels)}")

    train_labels, test_labels = map(
        lambda x: Tensor(x, requires_grad=True), [train_labels, test_labels]
    )

    train_images = Tensor(
        train_images.reshape(-1, 28 * 28) / 255, requires_grad=True
    ).float()
    test_images = Tensor(
        test_images.reshape(-1, 28 * 28) / 255, requires_grad=True
    ).float()

    model = Network()
    optimizer = Adam(model.parameters(), lr=LR)

    assert train_images.shape[1] == 28 * 28, "Invalid image shape"

    start_time = time.perf_counter()
    train(model, optimizer, train_images, train_labels)
    print(f"Time to train: {time.perf_counter() - start_time} seconds")
