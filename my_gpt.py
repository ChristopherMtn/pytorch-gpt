from enum import Enum, auto
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1134)

context_length = 512
batch_size = 24
d_model = 384
num_heads = 6
num_layers = 5
device = "mps" if torch.backends.mps.is_available() else "cpu"
learning_rate = 3e-4
max_iters = 4000
eval_interval = 200

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(int_list: list[int]) -> str:
    return ''.join([itos[i] for i in int_list])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train = data[:n]
validate = data[n:]

class SplitType(Enum):
    train = auto()
    validate = auto()
    test = auto()

split_to_data = {SplitType.train: train, SplitType.validate: validate, SplitType.test: []}

def get_batch(split_type: SplitType = SplitType.train):
    """
    Given a split type, queries the split to get
    a training batch and a target batch.
    """
    data = split_to_data[split_type]
    ix = torch.randint(len(data) - context_length - 1, (batch_size,))
    iy = ix+1
    x = torch.stack([data[i:i+context_length] for i in ix], dim=0)
    y = torch.stack([data[i:i+context_length] for i in iy])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    eval_iters = 100
    for split in [SplitType.train, SplitType.validate]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class MaskedSelfAttention(nn.Module):
    def __init__(self, head_size, d_model, context_length):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, hs)
        k = self.key(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)
        qkt = q @ k.transpose(1, 2) / self.head_size**0.5 # (B, T, T)
        qkt = qkt.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T) masked
        softmax_qkt = torch.nn.functional.softmax(qkt, -1)  # (B, T, T)
        result = softmax_qkt @ v  # (B, T, hs)
        return result  # (B, T, hs)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, context_length):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        head_size = d_model // num_heads
        self.heads = nn.ModuleList([MaskedSelfAttention(head_size, d_model, context_length) for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        res = self.proj(x)
        return res

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fully_connected_1 = torch.nn.Linear(d_model, 4*d_model)
        self.relu = torch.nn.ReLU()
        self.fully_connected_2 = torch.nn.Linear(4*d_model, d_model)

    def forward(self, x):
        x = self.fully_connected_1(x)
        x = self.relu(x)
        x = self.fully_connected_2(x)
        return x

class Layer(nn.Module):
    def __init__(self, d_model, num_heads, context_length):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, context_length)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm_1(x))
        x = x + self.ffwd(self.layer_norm_2(x))
        return x

class LLM(nn.Module):
    def __init__(self, d_model, num_heads, context_length):
        # vocab_size->vocab_size means this works like a one-hot vector
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(vocab_size, d_model)
        self.positional_embedding = torch.nn.Embedding(context_length, d_model)
        self.layers = torch.nn.Sequential(*[Layer(d_model, num_heads, context_length) for _ in range(num_layers)])
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.embed_to_vocab = torch.nn.Linear(d_model, vocab_size)

    def forward(self, inputs, targets=None):
        # input = (B, T)
        B, T = inputs.shape
        assert T <= context_length, f'time dim ({T}) is larger than supported context length ({context_length})'  # Need to check this on MPS backend
        token_embeddings = self.token_embeddings(inputs)  # (B, T, n_embed)
        positional_embeddings = self.positional_embedding(torch.arange(T, device=device))  # (B, T, n_embed)
        x = token_embeddings + positional_embeddings  # (B, T, n_embed)
        x = self.layers(x)  # (B, T, n_embed)
        x = self.layer_norm(x)  # Deviating from Attention is All You Need, adding LNs before heads and FF. Final LN here.
        logits = self.embed_to_vocab(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            flattened_logits = logits.view(B*T, C)
            flattened_targets = targets.view(B*T)
            loss = F.cross_entropy(flattened_logits, flattened_targets)

        return logits, loss

    def generate(self, context, max_new_tokens=3000):
        # context is (1, T) array of indices
        for _ in range(max_new_tokens):
            assert len(context.shape) == 2 and context.shape[0] == 1, 'Bad context shape'
            cropped_context = context[:, -context_length:]
            logits, _ = self.forward(cropped_context)
            last_logits = logits[0, -1]
            probs = F.softmax(last_logits, dim=-1)
            next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next.view(1, 1)), dim=1)
        return context

model = LLM(d_model, num_heads, context_length)
model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train Loop
for iter in range(max_iters + 1):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses[SplitType.train]:.4f}, val loss {losses[SplitType.validate]:.4f}")

    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# Generate some text using our trained model
torch.save(model.state_dict(), 'shakespearer.pt')
print("Model weights saved!")

print('generating...')
print('\n')
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generation = decode(model.generate(context)[0].tolist())
print(generation)