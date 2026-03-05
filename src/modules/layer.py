import torch.nn as nn
from src.modules.rms_norm import RMSNorm
from src.modules.group_query_attention import GroupQueryAttention
from src.modules.moe import MOE

class Layer(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, context_length, num_experts, active_experts):
        super().__init__()
        self.rms_norm_1 = RMSNorm(d_model)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        head_size = d_model // num_heads
        self.self_attn = GroupQueryAttention(d_model, head_size, num_heads, num_kv_heads, context_length)
        self.rms_norm_2 = RMSNorm(d_model)
        self.ffwd = MOE(num_experts, active_experts, d_model)

    def forward(self, x):
        x = x + self.self_attn(self.rms_norm_1(x))
        x = x + self.ffwd(self.rms_norm_2(x))
        return x