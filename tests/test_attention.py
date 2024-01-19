from punytorch.tensor import Tensor
from examples.gpt import MHA, MLP, ModelArgs
import numpy as np

# Initialize our models
model_args = ModelArgs(
    seq_len=10, d_model=16, n_heads=2, vocab_size=10, num_layers=2, esp=0.1
)
multi_head_attention = MHA(model_args)
multi_layer_perceptron = MLP(in_features=16, out_features=16, expansion_size=3)


def test_output_shape_attention():
    # Test the output shape of the multi-head attention
    input = Tensor(np.random.randn(32, model_args.seq_len, model_args.d_model))
    output = multi_head_attention(input)
    assert output.shape == input.shape


def test_output_shape_mlp():
    # Test the output shape of the multi-layer perceptron
    input = Tensor(np.random.randn(32, 16))
    output = multi_layer_perceptron(input)
    assert output.shape == input.shape


# more tests to come
