"""
gpt.py

This example implements a GPT model for text generation using only punytorch (and numpy).

Table of Contents:
1. Imports
2. Dataclasses for Model and Hyperparameters
3. Helper Functions
4. Model Components (MHA, MLP, RMSNorm, Block, GPT)
5. Main Function
"""

# 1. Imports
from dataclasses import dataclass
import logging
import numpy as np

from tqdm import tqdm

from punytorch.activations import ReLU, Softmax
from punytorch.helpers import CharTokenizer
from punytorch.losses import CrossEntropyLoss
import punytorch.nn as nn
from punytorch.optim import Adam
from punytorch.tensor import Tensor


# 2. Dataclasses for Model and Hyperparameters
@dataclass
class Hyperparameters:
    batch_size: int
    block_size: int
    max_iters: int
    eval_interval: int
    learning_rate: float
    device: str
    eval_iters: int
    num_embeds: int
    num_heads: int
    num_layers: int
    dropout: float


# 3. Helper Functions
@Tensor.no_grad()
def estimate_loss(model, train_data, val_data, hparams: Hyperparameters):
    """
    Estimates the loss of the model over a number of iterations.

    This function runs the model in evaluation mode and computes the average loss over a specified number of iterations.
    The loss is computed separately for the training and validation sets.

    Args:
        model (Module): The model to evaluate.
        train_data: The training dataset.
        val_data: The validation dataset.
        hyperparameters: The hyperparameters of the model.

    Returns:
        dict: A dictionary with the average loss for the training and validation sets.
    """
    out = {}

    # put the model in evaluation mode
    # - prevents neuron dropout
    # - batchnorm layers use running statistics instead of batch statistics
    model.eval()

    for split in ["train", "val"]:
        losses = []
        for k in range(hparams.eval_iters):
            data, targets = get_batch(split, train_data, val_data, hparams)
            logits = model(data)

            batch_size, time_step, channels = logits.shape
            logits = logits.view(batch_size * time_step, channels)
            targets = targets.view(batch_size * time_step)
            loss = CrossEntropyLoss.forward(logits, targets)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)

    # return the model to training mode
    # it's generally good practice to make sure that a function cleans up after itself
    model.train()
    return out


def get_batch(split, train_data, val_data, hparams: Hyperparameters):
    """
    Generates a batch of data for training or validation.

    Args:
        split (str): Specifies whether the batch is for training or validation.
                     Should be either "train" or "val".
        train_data (np.ndarray): The training dataset.
        val_data (np.ndarray): The validation dataset.
        block_size (int): The size of each block of data.
        batch_size (int): The number of data blocks in each batch.
        device (str): The device to which the data should be sent.
                      Should be either "cpu" or "gpu".

    Returns:
        tuple: A tuple containing two Tensors. The first tensor contains the input data
               and the second tensor contains the target data.
    """
    # choose the correct dataset based on the 'split' parameter
    data = train_data if split == "train" else val_data
    len_data = len(data)

    # randomly select starting indices for the sequences
    idx = np.random.randint(0, len_data - hparams.block_size, hparams.batch_size)

    # create input (x) and target (y) sequences based on block_size
    # target (y) sequence is offset by one (common practice in language modeling)
    x = Tensor.stack([data[i : i + hparams.block_size] for i in idx])
    y = Tensor.stack([data[i + 1 : i + hparams.block_size + 1] for i in idx])

    # move the tensor to the specified device
    x, y = x.to(hparams.device), y.to(hparams.device)
    return x, y


@Tensor.no_grad()
def generate(model, idx, max_new_tokens, hparams: Hyperparameters):
    """
    Generates new tokens using the trained model.

    This function takes a starting index tensor (usually zeros) and generates a sequence of tokens
    by sampling from the probability distribution output by the model. It continues generating tokens
    until it reaches the specified number of max_new_tokens.

    Args:
        model (Module): The trained GPT model used for generation.
        idx (Tensor): The initial index tensor, typically zeros, used as a starting point for generation.
        max_new_tokens (int): The maximum number of new tokens to generate.
        hyperparameters: The hyperparameters of the model, containing settings like block_size and device.

    Returns:
        Tensor: A tensor containing the indices of the generated tokens.
    """
    idx = Tensor.zeros((1, hparams.block_size)).to(hparams.device).long()
    for i in range(max_new_tokens):
        idx_cond = idx[:, -hparams.block_size :]
        logits = model(idx_cond)
        logits = logits[:, -1, :]  # only take the last token, since we're predicting the "next" token

        # logits are converted to a probability distribution using the softmax function
        probs = Softmax().forward(logits, dim=-1)

        # the next index is sampled from the probability distribution, then added to the sequence
        idx_next = Tensor.multinomial(probs, num_samples=1)
        idx = Tensor.cat((idx, idx_next), dim=1)

    # return the model to training mode
    model.train()
    return idx[:, hparams.block_size :]


# 4. Model Components (Attention (Single head and MHA), MLP, RMSNorm, Block, GPT)
class Head(nn.Module):
    """
    A single attention head.

    This class implements a single attention head, which is a key component of the transformer architecture.
    It computes the attention scores, applies a mask, and performs the attention operation.

    Args:
        head_size (int): The size of the attention head.
        hyperparameters (Hyperparameters): The hyperparameters of the model.
    """

    def __init__(self, head_size, hparams: Hyperparameters):
        super().__init__()
        self.key = nn.Linear(hparams.num_embeds, head_size)
        self.query = nn.Linear(hparams.num_embeds, head_size)
        self.value = nn.Linear(hparams.num_embeds, head_size)
        self.register_buffer("tril", Tensor(np.tril(np.ones((hparams.block_size, hparams.block_size)))))
        self.dropout = nn.Dropout(hparams.dropout)

    def forward(self, x):
        """
        Computes the forward pass of the attention head.

        Args:
            x (Tensor): The input tensor of shape (batch_size, sequence_length, num_embeds).

        Returns:
            Tensor: The output tensor after applying attention, of shape (batch_size, sequence_length, head_size).
        """
        batch_size, sequence_length, channels = x.shape

        # Compute key, query, and value projections
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Compute attention scores
        attention_scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5

        # Apply mask to attention scores
        masked_attention_scores = attention_scores.masked_fill(
            self.tril[:sequence_length, :sequence_length] == 0, float("-inf")
        )

        # Compute attention probabilities
        attention_probs = Softmax().forward(masked_attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Compute the attended values
        v = self.value(x)
        out = attention_probs @ v

        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.

    This module applies multiple attention heads in parallel and concatenates their outputs.
    The concatenated output is then projected to the original embedding dimension.

    Args:
        num_heads (int): The number of attention heads.
        head_size (int): The size of each attention head.
        hyperparameters (Hyperparameters): The hyperparameters of the model.
    """

    def __init__(self, head_size, hparams: Hyperparameters):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, hparams) for _ in range(hparams.num_heads)])
        self.proj = nn.Linear(hparams.num_embeds, hparams.num_embeds)
        self.dropout = nn.Dropout(hparams.dropout)

    def forward(self, x):
        """
        Computes the forward pass of the multi-head attention module.

        Args:
            x (Tensor): The input tensor of shape (batch_size, sequence_length, num_embeds).

        Returns:
            Tensor: The output tensor after applying multi-head attention, of shape (batch_size, sequence_length, num_embeds).
        """
        # Apply attention heads in parallel
        head_outputs = [h(x) for h in self.heads]

        # Concatenate the outputs of all attention heads
        concatenated = Tensor.cat(head_outputs, dim=-1)

        # Project the concatenated output back to the original embedding dimension
        out = self.proj(concatenated)

        # Apply dropout regularization
        out = self.dropout(out)

        return out


class MLP(nn.Module):
    def __init__(self, in_features, out_features, expansion_size: int = 3):
        """
        Initializes the MLP module.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            expansion_size (int, optional): The factor by which the input features are expanded in the hidden layers. Defaults to 3.
        """
        super().__init__()
        self.w1 = nn.Linear(in_features, in_features * expansion_size)
        self.w2 = nn.Linear(in_features * expansion_size, in_features * expansion_size)
        self.w3 = nn.Linear(in_features * expansion_size, out_features)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the MLP.
        """
        x = self.w1(x)
        x = ReLU().forward(x)
        x = self.w2(x)
        x = ReLU().forward(x)
        x = self.w3(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm module.

        Args:
            dim (int): The dimension over which to compute the root mean square.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-5.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(np.ones(dim))

    def _norm(self, x: Tensor):
        """
        Computes the root mean square of the input tensor along the last dimension.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The normalized tensor.
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x to be a Tensor, but got {type(x).__name__}")
        rms = ((x**2).mean(axis=-1, keepdims=True) + self.eps) ** 0.5
        normalized_x = x / rms
        if not isinstance(normalized_x, Tensor):
            raise TypeError(f"Expected normalized_x to be a Tensor, but got {type(normalized_x).__name__}")
        return normalized_x

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the RMSNorm.
        """
        output = self._norm(x)
        if not isinstance(self.weight, Tensor):
            raise TypeError(f"Expected self.weight to be a Tensor, but got {type(self.weight)}")
        return output * self.weight


class Block(nn.Module):
    """
    A Block represents a single block in the transformer model, consisting of a
    Multi-Head Attention (MHA) layer and a Feed-Forward Network (FFN).
    Each of these components has a residual connection around it, followed by a
    layer normalization.
    """

    def __init__(self, hparams: Hyperparameters) -> None:
        """
        Initializes the Block module.

        Args:
            model_args (ModelArgs): The arguments for the model, including dimensions and sequence length.
        """
        super().__init__()
        head_size = hparams.num_embeds // hparams.num_heads
        self.attn = MultiHeadAttention(head_size, hparams)
        self.ffn = MLP(in_features=hparams.num_embeds, out_features=hparams.num_embeds)
        self.l1 = RMSNorm(hparams.num_embeds, eps=hparams.dropout)
        self.l2 = RMSNorm(hparams.num_embeds, eps=hparams.dropout)

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


class GPT(nn.Module):
    """
    The GPT class represents the GPT model, which consists of token embeddings, position embeddings,
    a list of transformer blocks, a normalization layer, and a projection layer.
    """

    def __init__(self, hparams: Hyperparameters, vocab_size) -> None:
        """
        Initializes the GPT model.

        Args:
            model_args (ModelArgs): The arguments for the model, including dimensions and sequence length.
            device (str): The device to run the model on ("cpu" or "gpu").
        """
        super().__init__()
        self.device = hparams.device
        self.token_embedding = nn.Embedding(vocab_size, hparams.num_embeds)
        self.position_embedding = nn.Embedding(hparams.block_size, hparams.num_embeds)
        self.layers = nn.ModuleList([Block(hparams=hparams) for _ in range(hparams.num_layers)])
        self.norm = RMSNorm(hparams.num_embeds)
        self.proj = nn.Linear(hparams.num_embeds, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x: Tensor, targets=None) -> Tensor:
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
            targets (Tensor, optional): The target values. If provided, the method will compute and return the loss. If not provided, the method will only return the logits. Defaults to None.

        Returns:
            Tensor: The output of the GPT model.
        """
            raise TypeError(f"Expected x to be a Tensor, but got {type(x).__name__}")
        B, T = x.shape
        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(Tensor(np.arange(T)).to(self.device))
        x = token_embedding + position_embedding  # (B,T,C)
        x = self.layers(x)  # (B,T,C)
        x = self.norm(x)  # (B,T,C)
        logits = self.proj(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = CrossEntropyLoss.forward(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, hparams: Hyperparameters):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -hparams.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = Softmax.forward(logits, dim=-1)  # (B, C)
            # sample from the distribution - don't have access to torch.multinomial, so we're making our own
            idx_next = np.array([np.random.choice(len(probs[b]), 1, p=probs[b]) for b in range(len(probs))])  # (B, 1)
            idx_next = Tensor(idx_next).to(probs.device)  # Convert to Tensor and move to the same device as probs
            # append sampled index to the running sequence
            idx = Tensor.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# 5. Main Function
def main():
    # hyperparameters for the model and training run
    hyperparameters = Hyperparameters(
        batch_size=64,
        block_size=256,
        max_iters=5000,
        eval_interval=500,
        learning_rate=3e-4,
        device="cpu",
        eval_iters=200,
        num_embeds=384,
        num_heads=6,
        num_layers=6,
        dropout=0.2,
    )

    tokenizer = CharTokenizer(filepath="datasets/input.txt")

    data = Tensor(tokenizer.encode(tokenizer.text)).long()
    n = int(0.95 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    vocab_size = tokenizer.get_vocab_size()

    model = GPT(hparams=hyperparameters, vocab_size=vocab_size).to(hyperparameters.device)
    optimizer = Adam(model.parameters(), lr=hyperparameters.learning_rate)

    # type checks before moving on
    if not all(isinstance(x, Tensor) for x in [data, train_data, val_data]):
        raise TypeError("All data must be Tensors")
    if not all(isinstance(p, Tensor) for p in model.parameters()):
        raise TypeError("All model parameters must be instances of Tensor")

    # TODO:
    # 1. loop through hypermarameters.max_iters
    #    at each iteration, check if we are at the eval_iterval
    #    if yes, print the loss and generate some text
    #    if no, train the model
    # 2. get a batch of training data and targets using get_batch function
    # 3. feed the data to the model to get the model's predictions (logits)
    # 4. reshape the logits and targets to be two-dimensional
    #    the first dimension is the batch size times the sequence length
    #    the second dimension is the number of classes
    # 5. compute the loss using the logits and targets
    # 6. backpropagate the loss and update the model's parameters
    # 7. update model parameters using gradients and the optimizer's step function
    # 8. optimizer.zero_grad()
    # 9. if the current iteration is a multiple of 50, print current iteration and the loss
    #
    # after loop:
    # - create another context tensor
    # - generate some new tokens
    # - decode these tokens into text
    # - print the text
    with tqdm(total=hyperparameters.max_iters, desc="Training Progress") as pbar:
        for iter in range(1, hyperparameters.max_iters):
            if iter % hyperparameters.eval_interval == 0 or iter == hyperparameters.max_iters - 1:
                print("=" * 50)
                losses = estimate_loss(model, train_data, val_data, hyperparameters.eval_iters)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                context = Tensor(np.zeros((1, 1))).to(hyperparameters.device)
                print(tokenizer.decode(generate(model, context, max_new_tokens=500)[0].tolist()))
                optimizer.zero_grad()
                print("-" * 50)
            data, targets = get_batch("train", train_data, val_data, hyperparameters)
            logits = model(data)
            batch_size, time_step, channels = logits.shape
            logits = logits.view(batch_size * time_step, channels)
            targets = targets.view(batch_size * time_step)
            loss = CrossEntropyLoss.forward(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if iter % 50 == 0:
                print(f"{iter=} {loss.item()=}")

            pbar.update(1)
    context = Tensor.zeros((1, 1)).to(hyperparameters.device).long()
    print(tokenizer.decode(generate(model, context, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG")
    logger = logging.getLogger(__name__)
    main()