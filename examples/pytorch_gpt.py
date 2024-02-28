from dataclasses import dataclass
import os
import requests

import torch
import torch.nn as nn
from torch.nn import functional as F

from examples.gpt import get_batch, estimate_loss


@dataclass
class Hyperparameters:
    batch_size: int
    block_size: int
    max_iters: int
    eval_interval: int
    learning_rate: float
    device: str
    eval_iters: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float


torch.manual_seed(1337)


# tiny shakespeare dataset stuff
dataset_path = "datasets/input.txt"
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not os.path.exists(dataset_path):
    print(f"{dataset_path} not found, downloading from {url}")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    response = requests.get(url)
    with open(dataset_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"Downloaded and saved to {dataset_path}")
else:
    print(f"{dataset_path} already exists.")

with open(dataset_path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
string_to_int = {char: int for int, char in enumerate(chars)}
int_to_string = {int: char for int, char in enumerate(chars)}
encode = lambda string: [string_to_int[char] for char in string]
decode = lambda lyst: "".join([int_to_string[int] for int in lyst])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split, hyperparameters):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - hyperparameters.block_size, (hyperparameters.batch_size,))
    x = torch.stack([data[i : i + hyperparameters.block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + hyperparameters.block_size + 1] for i in idx])
    x, y = x.to(hyperparameters.device), y.to(hyperparameters.device)
    return x, y


@torch.no_grad()
def estimate_loss(model, hyperparameters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(hyperparameters.eval_iters)
        for k in range(hyperparameters.eval_iters):
            X, Y = get_batch(split, hyperparameters)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, hyperparameters):
        super().__init__()
        self.key = nn.Linear(hyperparameters.n_embd, head_size, bias=False)
        self.query = nn.Linear(hyperparameters.n_embd, head_size, bias=False)
        self.value = nn.Linear(hyperparameters.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(hyperparameters.block_size, hyperparameters.block_size)))

        self.dropout = nn.Dropout(hyperparameters.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, hyperparameters):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, hyperparameters) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, hyperparameters.n_embd)
        self.dropout = nn.Dropout(hyperparameters.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, hyperparameters):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hyperparameters.n_embd, 4 * hyperparameters.n_embd),
            nn.ReLU(),
            nn.Linear(4 * hyperparameters.n_embd, hyperparameters.n_embd),
            nn.Dropout(hyperparameters.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, hyperparameters):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = hyperparameters.n_embd // hyperparameters.n_head
        self.sa = MultiHeadAttention(hyperparameters.n_head, head_size, hyperparameters)
        self.ffwd = FeedFoward(hyperparameters)
        self.ln1 = nn.LayerNorm(hyperparameters.n_embd)
        self.ln2 = nn.LayerNorm(hyperparameters.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, hyperparameters.n_embd)
        self.position_embedding_table = nn.Embedding(hyperparameters.block_size, hyperparameters.n_embd)
        self.blocks = nn.Sequential(*[Block(hyperparameters) for _ in range(hyperparameters.n_layer)])
        self.ln_f = nn.LayerNorm(hyperparameters.n_embd)  # final layer norm
        self.lm_head = nn.Linear(hyperparameters.n_embd, vocab_size)
        self.device = hyperparameters.device

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, hyperparameters):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -hyperparameters.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def main():
    import time  # Import the time module to measure training duration

    hyperparameters = Hyperparameters(
        batch_size=64,
        block_size=256,
        max_iters=5000,
        eval_interval=500,
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_iters=200,
        n_embd=384,
        n_head=6,
        n_layer=6,
        dropout=0.2,
    )
    model = GPTLanguageModel(hyperparameters)
    m = model.to(hyperparameters.device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters.learning_rate)

    total_tokens = len(encode(text))  # Assuming 'text' contains the entire dataset
    train_tokens = int(0.9 * total_tokens)  # 90% for training
    block_size = hyperparameters.block_size  # From the Hyperparameters data class

    # Calculate the maximum number of non-overlapping sequences
    max_sequences = train_tokens // block_size

    print(f"Maximum number of non-overlapping training sequences: {max_sequences}")

    start_time = time.time()  # Record the start time of the training

    for iter in range(hyperparameters.max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % hyperparameters.eval_interval == 0 or iter == hyperparameters.max_iters - 1:
            current_time = time.time()  # Get the current time
            elapsed_time = current_time - start_time  # Calculate elapsed time since the start
            losses = estimate_loss(model, hyperparameters)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}. Elapsed time: {elapsed_time:.2f} seconds"
            )

        # sample a batch of data
        xb, yb = get_batch("train", hyperparameters)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_training_time = time.time() - start_time  # Calculate total training time
    print(f"Total training time: {total_training_time:.2f} seconds")  # Print total training time
    epochs = (max_sequences / hyperparameters.batch_size) / hyperparameters.max_iters
    time_per_epoch = total_training_time / epochs
    print(f"Total epochs: {epochs}, Time per epoch: {time_per_epoch} seconds")

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=hyperparameters.device)
    print(decode(m.generate(context, max_new_tokens=500, hyperparameters=hyperparameters)[0].tolist()))
    # open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))


if __name__ == "__main__":
    main()
