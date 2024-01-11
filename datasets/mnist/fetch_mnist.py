import gzip
import os
import struct

import numpy as np
import requests


def download_mnist(directory):
    base_url = "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    save_dir = directory
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


def load_mnist(directory) -> tuple:
    def read_labels(filename: str) -> np.array:
        with gzip.open(filename, "rb") as f:
            magic = struct.unpack(">I", f.read(4))
            if magic[0] != 2049:
                raise ValueError(
                    f"Invalid magic number {magic}, aborting read of labels."
                )
            num_items = struct.unpack(">I", f.read(4))[0]
            return np.frombuffer(f.read(), dtype=np.uint8, count=num_items)

    def read_images(filename: str) -> np.array:
        with gzip.open(filename, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(
                    f"Invalid magic number {magic}, aborting read of images."
                )
            images = np.frombuffer(f.read(), dtype=np.uint8)
            return images.reshape(num, rows, cols, 1)

    train_labels = read_labels(f"{directory}/train-labels-idx1-ubyte.gz")
    train_images = read_images(f"{directory}/train-images-idx3-ubyte.gz")
    test_labels = read_labels(f"{directory}/t10k-labels-idx1-ubyte.gz")
    test_images = read_images(f"{directory}/t10k-images-idx3-ubyte.gz")

    return (train_images, train_labels), (test_images, test_labels)
