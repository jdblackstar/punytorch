import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from datasets.mnist.fetch_mnist import download_mnist, load_mnist


def visualize_mnist(images: np.array, labels: np.array, grid_size=5):
    # going to add some randomness here to get a random slice of data each time
    indices = np.random.choice(len(images), grid_size**2, replace=False)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=((grid_size * 2), (grid_size * 2)))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[indices[i]], cmap="gray")
        ax.set_title(f"Label: {labels[indices[i]]}")
        ax.axis("off")
    plt.show()


def main(grid_size: int):
    MNIST_DIR = "datasets/mnist"
    if not os.path.exists(MNIST_DIR) or not os.listdir(MNIST_DIR):
        download_mnist(MNIST_DIR)
    (train_images, train_labels), _ = load_mnist(MNIST_DIR)

    assert train_images is not None and train_labels is not None, "Failed to load images or labels."
    assert len(train_images) == len(train_labels), "Mismatch between number of images and labels."

    visualize_mnist(train_images, train_labels, grid_size=grid_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MNIST data.")
    parser.add_argument(
        "--grid_size",
        type=int,
        default=5,
        help="Sizes of the image grid to display. Will always be square.",
    )
    args = parser.parse_args()
    main(grid_size=args.grid_size)
