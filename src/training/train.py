import torch
from enum import Enum, auto
from src.modules.llm import LLM
from src.modules.moe import MOE
from src.training.tokenizer import Tokenizer

torch.manual_seed(1134)
torch.set_float32_matmul_precision('high')  # Apparently speeds up Apple silicon

context_length = 128
batch_size = 20
d_model = 448
num_heads = 8
num_layers = 6
num_experts = 2
active_experts = 1
num_kv_heads = 2
device = "mps" if torch.backends.mps.is_available() else "cpu"
learning_rate = 3e-4
max_iters = 5000
eval_interval = 200

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = Tokenizer(text)
vocab_size = tokenizer.vocab_size

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
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

def collect_aux_loss(model):
    aux_loss = 0.0
    for module in model.modules():
        if isinstance(module, MOE) and hasattr(module, 'aux_loss'):
            aux_loss = aux_loss + module.aux_loss
    return aux_loss

model = LLM(d_model, num_heads, num_kv_heads, context_length, num_experts, active_experts, vocab_size, num_layers)
model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train Loop
for iter in range(max_iters + 1):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        aux = collect_aux_loss(model)
        print(f"step {iter}: train loss {losses[SplitType.train]:.4f}, val loss {losses[SplitType.validate]:.4f}, aux loss {aux:.4f}")

    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    aux_loss = collect_aux_loss(model)
    total_loss = loss + 0.01 * aux_loss
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()


# Generate some text using our trained model
model_weights_name = 'shakespearer.pt'
torch.save(model.state_dict(), model_weights_name)
print(f"Model weights saved as {model_weights_name}!")

# Generate some text using our trained model
torch.save(model.state_dict(), 'shakespearer.pt')
print("Model weights saved!")

print('generating...')
print('\n')
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generation = tokenizer.decode(model.generate(context)[0].tolist())
print(generation)