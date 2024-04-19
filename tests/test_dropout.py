import numpy as np
from punytorch.nn.dropout import Dropout
from punytorch.tensor import Tensor


def test_dropout():
    np.random.seed(42)  # For reproducibility
    input_tensor = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
    dropout_layer = Dropout(p=0.5)

    # Test during training
    output_tensor_train = dropout_layer(input_tensor, train=True)
    assert (
        output_tensor_train.shape == input_tensor.shape
    ), "Output tensor shape should match input tensor shape during training."

    # Test during evaluation
    output_tensor_eval = dropout_layer(input_tensor, train=False)
    assert np.array_equal(
        output_tensor_eval.data, input_tensor.data
    ), "Output tensor should match input tensor during evaluation."

    # Check if dropout is applied (not a rigorous test due to randomness, but a basic check)
    assert not np.array_equal(
        output_tensor_train.data, input_tensor.data
    ), "Output tensor should not match input tensor during training (with dropout applied)."
