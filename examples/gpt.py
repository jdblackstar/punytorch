# import punytorch as torch
from punytorch.tensor import Tensor
from punytorch.nn.modules import Module, Linear, Embedding, Parameter, ModuleList
from punytorch.activations import Softmax, ReLU
from punytorch.losses import CrossEntropyLoss
from punytorch.nn.optimizers import Adam

# I think pytorch.nn.function is the same as our optimizers file
# import punytorch.nn.functional as F  # TODO: IMPLEMENT this
import punytorch.nn.optimizers as optim

from dataclasses import dataclass
import numpy as np
import math
from tqdm import tqdm


@dataclass
class Hyperparameters:
    batch_size: int
    block_size: int
    max_iters: int
    eval_interval: int
    learning_rate: float
    device: str = "cpu"
    eval_iters: int
    num_embeds: int
    num_heads: int
    num_layers: int
    dropout: float


@dataclass
class ModelArgs:
    seq_len: int
    d_model: int
    n_heads: int
    vocab_size: int
    num_layers: int
    esp: float


class MHA(Module):
    def __init__(self, model_args: ModelArgs) -> None:
        """
        Initializes the Multi-Head Attention module.

        Args:
            model_args (ModelArgs): The arguments for the model, including dimensions and sequence length.
        """
        super().__init__()
        self.key = Linear(model_args.d_model, model_args.d_model)
        self.query = Linear(model_args.d_model, model_args.d_model)
        self.value = Linear(model_args.d_model, model_args.d_model)
        self.proj = Linear(model_args.d_model, model_args.d_model)
        self.head_dim = model_args.d_model // model_args.n_heads

        self.n_heads = model_args.n_heads
        mask = Tensor(
            (
                np.tril(np.zeros((1, 1, model_args.seq_len, model_args.seq_len)))
                + np.triu(
                    -np.inf * np.ones((1, 1, model_args.seq_len, model_args.seq_len)),
                    k=1,
                )
            )
        ).float()

        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the Multi-Head Attention layer.
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        k = k.reshape(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.reshape(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        attn = self.attention(k, q, v, self.mask)
        v = attn.transpose(1, 2).reshape(B, T, C)
        x = self.proj(v)
        return x

    @staticmethod
    def attention(k, q, v, mask) -> Tensor:
        """
        Computes the attention scores.

        Args:
            k (Tensor): The key vectors.
            q (Tensor): The query vectors.
            v (Tensor): The value vectors.
            mask (Tensor): The mask tensor.

        Returns:
            Tensor: The output of the attention mechanism.
        """
        B, n_head, T, C = k.shape
        wei = (q @ k.transpose(-1, -2)) * (C**-0.5)
        wei = mask[:, :, :T, :T] + wei
        wei = Softmax(wei, dim=-1)
        x = wei @ v
        return x


class MLP(Module):
    def __init__(self, in_features, out_features, expansion_size: int = 3):
        """
        Initializes the MLP module.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            expansion_size (int, optional): The factor by which the input features are expanded in the hidden layers. Defaults to 3.
        """
        super().__init__()
        self.w1 = Linear(in_features, in_features * expansion_size)
        self.w2 = Linear(in_features * expansion_size, in_features * expansion_size)
        self.w3 = Linear(in_features * expansion_size, out_features)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the MLP.
        """
        x = self.w1(x)
        x = ReLU(x)
        x = self.w2(x)
        x = ReLU(x)
        x = self.w3(x)
        return x


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm module.

        Args:
            dim (int): The dimension over which to compute the root mean square.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-5.
        """
        super().__init__()
        self.eps = eps
        self.weight = Parameter(Tensor(np.ones(dim)))

    def _norm(self, x: Tensor):
        """
        Computes the root mean square of the input tensor along the last dimension.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The normalized tensor.
        """
        rms = ((x**2).mean(axis=-1, keepdim=True) + self.eps) ** 0.5
        return x / rms

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the RMSNorm.
        """
        output = self._norm(x)
        return output * self.weight


class Block(Module):
    """
    A Block represents a single block in the transformer model, consisting of a
    Multi-Head Attention (MHA) layer and a Feed-Forward Network (FFN).
    Each of these components has a residual connection around it, followed by a
    layer normalization.
    """

    def __init__(self, model_args: ModelArgs) -> None:
        """
        Initializes the Block module.

        Args:
            model_args (ModelArgs): The arguments for the model, including dimensions and sequence length.
        """
        super().__init__()
        self.attn = MHA(model_args)
        self.ffn = MLP(model_args.d_model, model_args.d_model)
        self.l1 = RMSNorm(model_args.d_model, eps=model_args.esp)
        self.l2 = RMSNorm(model_args.d_model, eps=model_args.esp)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the Block.
        """
        x = x + self.attn(self.l1(x))
        x = x + self.ffn(self.l2(x))
        return x


class GPT(Module):
    """
    The GPT class represents the GPT model, which consists of token embeddings, position embeddings,
    a list of transformer blocks, a normalization layer, and a projection layer.
    """

    def __init__(self, model_args: ModelArgs, device: str):
        """
        Initializes the GPT model.

        Args:
            model_args (ModelArgs): The arguments for the model, including dimensions and sequence length.
            device (str): The device to run the model on ("cpu" or "gpu").
        """
        super().__init__()
        self.device = device
        self.token_embedding = Embedding(model_args.vocab_size, model_args.d_model)
        self.position_embedding = Embedding(model_args.seq_len, model_args.d_model)
        self.layers = ModuleList(
            [Block(model_args) for _ in range(model_args.num_layers)]
        )
        self.norm = RMSNorm(model_args.d_model)
        self.proj = Linear(model_args.d_model, model_args.vocab_size)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the GPT model.
        """
        B, T = x.shape

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(Tensor(np.arange(T)).to(self.device))
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.proj(x)

        return logits


@Tensor.no_grad()
def estimate_loss(model, eval_iters):
    """
    Estimates the loss of the model over a number of iterations.

    This function runs the model in evaluation mode and computes the average loss over a specified number of iterations.
    The loss is computed separately for the training and validation sets.

    Args:
        model (Module): The model to evaluate.
        eval_iters (int): The number of iterations to run for the evaluation.

    Returns:
        dict: A dictionary with the average loss for the training and validation sets.
    """
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = []
        for k in range(eval_iters):
            data, targets = get_batch(split)
            logits = model(data)

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = CrossEntropyLoss.forward(logits, targets)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


@Tensor.no_grad()
def get_batch():
    raise NotImplementedError


