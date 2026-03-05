import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.rms_norm import RMSNorm
from src.modules.layer import Layer

class LLM(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, context_length, num_experts, active_experts, vocab_size, num_layers):
        # vocab_size->vocab_size means this works like a one-hot vector
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = torch.nn.Embedding(vocab_size, d_model)
        self.layers = torch.nn.Sequential(*[Layer(d_model, num_heads, num_kv_heads, context_length, num_experts, active_experts) for _ in range(num_layers)])
        self.rms_norm = RMSNorm(d_model)
        self.embed_to_vocab = torch.nn.Linear(d_model, vocab_size)

    def forward(self, inputs, targets=None):
        # input = (B, T)
        B, T = inputs.shape
        assert T <= self.context_length, f'time dim ({T}) is larger than supported context length ({self.context_length})'  # Need to check this on MPS backend
        token_embeddings = self.token_embeddings(inputs)  # (B, T, n_embed)
        x = self.layers(token_embeddings)  # (B, T, n_embed)
        x = self.rms_norm(x)  # Deviating from Attention is All You Need, adding norms before heads and FF. Final norm here.
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
            cropped_context = context[:, -self.context_length:]
            logits, _ = self.forward(cropped_context)
            last_logits = logits[0, -1]
            probs = F.softmax(last_logits, dim=-1)
            next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next.view(1, 1)), dim=1)
        return context