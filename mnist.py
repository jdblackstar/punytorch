# Standard library imports
import gzip
import os
import random
import struct
import time

# Third-party imports
import numpy as np
import requests
from tqdm import tqdm

from nn.models import MLP
from nn.modules import Linear, Module
from nn.optimizers import Adam
from punytorch.helpers import is_one_hot
from punytorch.losses import CrossEntropyLoss
from punytorch.tensor import Tensor

# Constants
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.25
MNIST_DIR = "mnist"


def download_mnist():
    base_url = "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    save_dir = MNIST_DIR
    os.makedirs(save_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(save_dir, file)

        if os.path.exists(file_path):
            continue

        url = base_url + file
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {file}")
        else:
            print(
                f"Failed to download {file}. HTTP Response Code: {response.status_code}"
            )


def load_mnist() -> tuple:
    def read_labels(filename: str) -> np.array:
        with gzip.open(filename, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    def read_images(filename: str) -> np.array:
        with gzip.open(filename, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            return images.reshape(num, rows, cols, 1)

    train_labels = read_labels(f"{MNIST_DIR}/train-labels-idx1-ubyte.gz")
    train_images = read_images(f"{MNIST_DIR}/train-images-idx3-ubyte.gz")
    test_labels = read_labels(f"{MNIST_DIR}/t10k-labels-idx1-ubyte.gz")
    test_images = read_images(f"{MNIST_DIR}/t10k-images-idx3-ubyte.gz")

    return (train_images, train_labels), (test_images, test_labels)


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
        return Tensor.softmax(self.l3(x))


@Tensor.no_grad()
def test(model: Network, test_images: Tensor, test_labels: Tensor):
    preds = model.forward(test_images)
    pred_indices = np.argmax(preds, axis=-1)

    # Convert one-hot encoded labels to class indices
    test_labels = np.argmax(test_labels.to_numpy(), axis=-1)

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
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix({"loss": float(loss.item())})

        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
        test(model, test_images, test_labels)


if __name__ == "__main__":
    download_mnist()
    (train_images, train_labels), (test_images, test_labels) = load_mnist()

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

    start_time = time.perf_counter()
    train(model, optimizer, train_images, train_labels)
    print(f"Time to train: {time.perf_counter() - start_time} seconds")
