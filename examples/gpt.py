"""
gpt.py

This example implements a GPT model for text generation using only punytorch.

Table of Contents:
1. Imports
2. Dataclasses for Model and Hyperparameters
3. Helper Functions
4. Model Components (MHA, MLP, RMSNorm, Block, GPT)
5. Main Function
"""

# 1. Imports
from dataclasses import dataclass
import numpy as np

from tqdm import tqdm

from punytorch.activations import ReLU, Softmax
from punytorch.helpers import CharTokenizer
from punytorch.losses import CrossEntropyLoss
from punytorch.nn.modules import Embedding, Linear, Module, ModuleList, Parameter
from punytorch.nn.optimizers import Adam
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


@dataclass
class ModelArgs:
    seq_len: int
    d_model: int
    n_heads: int
    vocab_size: int
    num_layers: int
    esp: float


# 3. Helper Functions
@Tensor.no_grad()
def estimate_loss(model, train_data, val_data, hyperparameters):
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
    model.eval()

    for split in ["train", "val"]:
        losses = []
        for k in range(hyperparameters.eval_iters):
            data, targets = get_batch(split, train_data, val_data, hyperparameters)
            logits = model(data)

            batch_size, time_step, channels = logits.shape
            logits = logits.view(batch_size * time_step, channels)
            targets = targets.view(batch_size * time_step)
            loss = CrossEntropyLoss.forward(logits, targets)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def get_batch(split, train_data, val_data, hyperparameters):
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
    data = train_data if split == "train" else val_data
    len_data = len(data)
    ix = np.random.randint(0, len_data - hyperparameters.block_size, hyperparameters.batch_size)
    x = Tensor.stack([data[i : i + hyperparameters.block_size] for i in ix])
    y = Tensor.stack([data[i + 1 : i + hyperparameters.block_size + 1] for i in ix])
    x, y = x.to(hyperparameters.device), y.to(hyperparameters.device)
    return x, y


@Tensor.no_grad()
def generate(model, idx, max_new_tokens, hyperparameters):
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
    idx = Tensor.zeros((1, hyperparameters.block_size)).to(hyperparameters.device).long()
    for i in range(max_new_tokens):
        idx_cond = idx[:, -hyperparameters.block_size :]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = Softmax().forward(logits, dim=-1)
        idx_next = Tensor.multinomial(probs, num_samples=1)
        idx = Tensor.cat((idx, idx_next), dim=1)
    model.train()
    return idx[:, hyperparameters.block_size :]


# 4. Model Components (MHA, MLP, RMSNorm, Block, GPT)
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
        batch_size, time_step, channels = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        key = key.reshape(batch_size, time_step, self.n_heads, channels // self.n_heads).transpose(1, 2)
        query = query.reshape(batch_size, time_step, self.n_heads, channels // self.n_heads).transpose(1, 2)
        value = value.reshape(batch_size, time_step, self.n_heads, channels // self.n_heads).transpose(1, 2)

        attn = self.attention(key, query, value, self.mask)
        attn = attn.reshape(batch_size, -1).reshape(batch_size, time_step, channels)
        return self.proj(attn)

    @staticmethod
    def attention(key, query, value, mask) -> Tensor:
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
        batch_size, n_head, time_step, channels = key.shape
        attention_scores = (query @ key.transpose(-1, -2)) * (channels**-0.5)
        attention_scores = mask[:, :, :time_step, :time_step] + attention_scores
        attention_scores = Softmax().forward(attention_scores, dim=-1)
        x = attention_scores @ value
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
        x = ReLU().forward(x)
        x = self.w2(x)
        x = ReLU().forward(x)
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
        self.layers = ModuleList([Block(model_args) for _ in range(model_args.num_layers)])
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

        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(Tensor(np.arange(T)).to(self.device))
        x = token_embedding + position_embedding

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.proj(x)

        return logits


# 5. Main Function
def main():
    # hyperparameters and modelargs
    hyperparameters = Hyperparameters(
        batch_size=64,
        block_size=128,
        max_iters=5000,
        eval_interval=500,
        learning_rate=3e-4,
        device="cpu",
        eval_iters=100,
        num_embeds=128 * 2,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
    )

    # fmt: off
    model_args = ModelArgs(
        seq_len=10,
        d_model=16,
        n_heads=2,
        vocab_size=10,
        num_layers=2,
        esp=1e-5,
    )
    # fmt: on

    model = GPT(model_args, hyperparameters.device).to(hyperparameters.device)
    optimizer = Adam(model.parameters(), lr=hyperparameters.learning_rate)
    tokenizer = CharTokenizer(filepath="datasets/input.txt")

    data = Tensor(tokenizer.encode(tokenizer.text)).long()
    n = int(0.95 * len(data))
    train_data = data[:n]
    val_data = data[n:]

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
    context = Tensor.zeros((1, 1)).to(hyperparameters.device).long()
    print(tokenizer.decode(generate(model, context, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
