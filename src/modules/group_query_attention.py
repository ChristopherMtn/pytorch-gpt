import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, head_size, num_heads, num_kv_heads, context_length):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "GQA requires clean divisibility"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.groups = num_heads // num_kv_heads  # Q heads per KV head
        self.head_size = head_size

        self.query = nn.Linear(d_model, num_heads * self.head_size, bias=False)
        self.key = nn.Linear(d_model, num_kv_heads * self.head_size, bias=False)
        self.value = nn.Linear(d_model, num_kv_heads * self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        # RoPE
        base=10000
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_size, 2).float() / head_size)
        )
        t = torch.arange(context_length).float()
        freqs = torch.outer(t, inv_freq)  # (T, head_size/2)

        self.register_buffer("cos_cached", torch.cos(freqs))
        self.register_buffer("sin_cached", torch.sin(freqs))

    def apply_rotary_emb(self, x, T):
        """
        x: (B, n_heads, T, head_size)
        """
        cos = self.cos_cached[:T]  # (T, head_size/2)
        sin = self.sin_cached[:T]

        # reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,head_size/2)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # split into pairs
        x = x.view(*x.shape[:-1], self.head_size // 2, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]

        # rotate
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        x = torch.stack((rotated_x1, rotated_x2), dim=-1)
        return x.view(*x.shape[:-2], self.head_size)

    def forward(self, x):
        B, T, _ = x.shape

        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        # (B, num_heads, T, head_size)

        k = self.key(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)
        # (B, num_kv_heads, T, head_size)

        # Apply RoPE to Q and K
        q = self.apply_rotary_emb(q, T)
        k = self.apply_rotary_emb(k, T)

        # Expand K/V so each KV head broadcasts across its group of Q heads
        # (B, num_heads, T, head_size) via repeat_interleave
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)

        scale = self.head_size ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, num_heads, T, T)
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # (B, num_heads, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        return out