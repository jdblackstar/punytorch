import logging

import numpy as np
import pytest

from examples.gpt import MHA, MLP, ModelArgs
from punytorch.tensor import Tensor

logger = logging.getLogger()

# Initialize our models
model_args = ModelArgs(seq_len=10, d_model=16, n_heads=2, vocab_size=10, num_layers=2, esp=0.1)


@pytest.mark.gpt
def test_output_shape_attention():
    multi_head_attention = MHA(model_args)
    # Test the output shape of the multi-head attention
    input = Tensor(np.random.randn(32, model_args.seq_len, model_args.d_model))
    output = multi_head_attention(input)
    assert output.shape == input.shape


@pytest.mark.gpt
def test_output_shape_mlp():
    multi_layer_perceptron = MLP(in_features=16, out_features=16, expansion_size=3)
    # Test the output shape of the multi-layer perceptron
    input = Tensor(np.random.randn(32, 16))
    output = multi_layer_perceptron(input)
    assert output.shape == input.shape


# more tests to come
