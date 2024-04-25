import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import json
import random

# hyperparameters
input_size = output_size = 159
batch_size = 8  # how many independent sequences will we process in parallel?
block_size = 16
max_iters = 50000
eval_interval = 200
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 50
n_embd = 16
n_head = 1
n_layer = 5
dropout = 0.01
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("leaves_training_data.json", "r", encoding="utf-8") as f:
    data = torch.tensor(json.load(f))

n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data_loader = DataLoader(
    TensorDataset(data[:n]), batch_size=batch_size, shuffle=True
)
val_data_loader = DataLoader(
    TensorDataset(data[n:]), batch_size=batch_size, shuffle=True
)

iter_train_data_loader = iter(train_data_loader)
iter_val_data_loader = iter(val_data_loader)


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data_loader = train_data_loader if split == "train" else val_data_loader
    iter_data_loader = (
        iter_train_data_loader if split == "train" else iter_val_data_loader
    )
    try:
        batch = next(iter_data_loader)[0]
    except:
        iter_data_loader = iter(data_loader)
        batch = next(iter_data_loader)[0]
    i = random.randint(1, batch.shape[1] - block_size)
    data = batch[:, -i - block_size : -i, :]
    target = batch[:, -i - block_size + 1 : -i + 1 if i != 1 else None, :]
    data, target = data.to(device), target.to(device)
    return data, target


naive_val_loss = F.l1_loss(data[:n, :-1, -2], data[:n, 1:, -2])
naive_train_loss = F.l1_loss(data[n:, :-1, -2], data[n:, 1:, -2])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def naive_loss(logits, targets):
    # The loss should be the mse of all logits vs targets, so:
    shape = logits.shape
    error_sum = 0
    for i in range(shape[0]): # batch_index
        for t in range(shape[1]): # time_step
            error_sum += F.mse_loss(logits[i, t], targets[i, t])
    
    return error_sum / (shape[0]*shape[1])


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(x)
        return x


class GPTFinancialModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.embedding = nn.Linear(input_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.lm_head = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(4 * n_embd, output_size),
            nn.Dropout(dropout),
        )

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
        B, T, D = idx.shape
        input_emb = self.embedding(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T,n_emb)
        x = input_emb + pos_emb  # (B,T,n_emb)
        x = self.blocks(x)  # (B,T,n_emb)
        # x = self.ln_f(x) # (B,T,n_emb)
        logits = self.lm_head(x)  # (B, T, output_size)

        if targets is None:
            loss = None
        else:
            loss = F.l1_loss(logits[:, :, -2], targets[:, :, -2])

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

model = GPTFinancialModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iteration in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iteration % eval_interval == 0 or iteration == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
