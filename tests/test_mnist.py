from examples.mnist import load_mnist


def test_mnist():
    (train_images, train_labels), (test_images, test_labels) = load_mnist()

    assert train_images.shape == (60000, 28, 28, 1), "Unexpected shape for train_images"
    assert train_labels.shape == (60000,), "Unexpected shape for train_labels"
    assert test_images.shape == (10000, 28, 28, 1), "Unexpected shape for test_images"
    assert test_labels.shape == (10000,), "Unexpected shape for test_labels"
