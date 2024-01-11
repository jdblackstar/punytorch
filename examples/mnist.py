# Standard library imports
import random
import time

# Third-party imports
import numpy as np  # will remove this ASAP
from tqdm import tqdm

from datasets.mnist.fetch_mnist import download_mnist, load_mnist
from punytorch.helpers import is_one_hot
from punytorch.losses import CrossEntropyLoss
from punytorch.nn.modules import Linear, Module
from punytorch.nn.optimizers import Adam
from punytorch.tensor import Tensor

# Constants
EPOCHS = 1
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
        x = Tensor.relu(self.l1(x))
        x = Tensor.relu(self.l2(x))
        return self.l3(x)


@Tensor.no_grad()
def test(model: Network, test_images: Tensor, test_labels: Tensor):
    preds = model.forward(test_images)
    pred_indices = np.argmax(preds, axis=-1)

    # Convert one-hot encoded labels to class indices
    test_labels = np.argmax(Tensor.data_to_numpy(test_labels.data), axis=-1)

    correct = 0
    for p, t in zip(pred_indices.reshape(-1), test_labels.reshape(-1)):
        if p == t:
            correct += 1
    accuracy = correct / len(test_labels)
    print(f"Test accuracy: {accuracy:.2%}")


def cross_entropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return CrossEntropyLoss.forward(y_pred, y_true)


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
                batch_labels = np.argmax(batch_labels, axis=-1)

                optimizer.zero_grad()
                pred = model.forward(batch_images)
                loss = cross_entropy(pred, batch_labels)
                loss.backward()
                for param in model.parameters():
                    print(param.grad)
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix({"loss": float(loss.item())})

        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
        test(model, test_images, test_labels)


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

    train_labels, test_labels = map(Tensor, [train_labels, test_labels])

    train_images = Tensor(train_images.reshape(-1, 28 * 28) / 255).float()
    test_images = Tensor(test_images.reshape(-1, 28 * 28) / 255).float()

    model = Network()
    optimizer = Adam(model.parameters(), lr=LR)

    assert train_images.shape[1] == 28 * 28, "Invalid image shape"

    start_time = time.perf_counter()
    train(model, optimizer, train_images, train_labels)
    print(f"Time to train: {time.perf_counter() - start_time} seconds")
