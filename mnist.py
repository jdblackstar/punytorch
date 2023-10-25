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

import punytorch as puny

# Constants
EPOCHS = 1
BATCH_SIZE = 32
LR = 4e-3
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
