import os 
import glob
import json
import torch
import torch.nn as nn
from torch.nn import functional as F

# Load data
def load_json_files(directory):
    shared_filenames = sorted(glob.glob(os.path.join(directory, "*.json")))
    with open(shared_filenames[0], 'r') as f:
        data = json.load(f)
    stories = [x['story'] for x in data]
    text = "\n".join(stories)
    return text

text = load_json_files("/Users/amansingh/Personal/Machine-Learning-Scripts/TinyStories")

# Tokenize
def char_tokenize(text):
    chars = sorted(list(set(text)))
    ctoi = {ch:i for i,ch in enumerate(chars)}
    itoc = {i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [ctoi[c] for c in s]
    decode = lambda l: ''.join([itoc[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    return data, encode, decode, ctoi, itoc   

data, encode, decode, ctoi, itoc = char_tokenize(text)
vocab_size = len(ctoi)

def data_split(data):
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    block_size = 256
    return train_data, val_data, block_size

train_data, val_data, block_size = data_split(data)

batch_size = 64
torch.manual_seed(0)

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x, y

class MOELayer(nn.Module):
    def __init__(self, experts, gate, k=1):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.k = k

    def forward(self, input):
        input_squashed = input.view(-1, input.shape[-1])
        gate_logits = self.gate(input_squashed)
        weights, selected_experts = torch.topk(gate_logits, self.k, dim=1)
        weights = F.softmax(weights, dim=1, dtype=torch.float).type_as(input)
        results = torch.zeros_like(input_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            if batch_idx.numel() > 0:
                expert_out = expert(input_squashed[batch_idx])
                weight = weights[batch_idx, nth_expert].unsqueeze(-1)
                results[batch_idx] += weight * expert_out
        return results.view_as(input)

class FeedForwardExpert(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embed, n_head, num_experts=4):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embed // n_head)
        self.ffwd = MOELayer(
            experts=[FeedForwardExpert(n_embed) for _ in range(num_experts)],
            gate=nn.Linear(n_embed, num_experts, bias=False)
        )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embed = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_embed + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Hyperparameters
max_iters = 3000 
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Transformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
