import matplotlib.pyplot as plt
import numpy as np
import os

from datasets.mnist.fetch_mnist import download_mnist, load_mnist


def visualize_mnist(images: np.array, labels: np.array):
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(f"Label: {labels[i]}")
        ax.axis("off")
    plt.show()


def main():
    MNIST_DIR = "datasets/mnist"
    if not os.path.exists(MNIST_DIR) or not os.listdir(MNIST_DIR):
        download_mnist(MNIST_DIR)
    (train_images, train_labels), _ = load_mnist(MNIST_DIR)

    assert (
        train_images is not None and train_labels is not None
    ), "Failed to load images or labels."
    assert len(train_images) == len(
        train_labels
    ), "Mismatch between number of images and labels."

    visualize_mnist(train_images, train_labels)


if __name__ == "__main__":
    main()
