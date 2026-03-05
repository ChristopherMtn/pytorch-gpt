import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.feed_forward import FeedForward

class MOE(nn.Module):
    def __init__(self, num_experts, active_experts, d_model):
        super().__init__()
        self.active_experts = active_experts
        self.experts = nn.ModuleList([FeedForward(d_model) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x):
        #  x is (B, T, d_model)
        B, T, d_model = x.shape
        num_tokens = B * T

        # Pick experts, build per-token weight and index tensors
        expert_selections = self.router(x)  # (B, T, num_experts)
        weights, indices = torch.topk(expert_selections, self.active_experts, dim=-1)  # both are dim (B, T, active_experts)
        weights = F.softmax(weights, dim=-1)

        # Aux Loss
        # Gradient descent tends to starve, so need to add loss that punishes
        # always choosing the same expert
        one_hot = torch.zeros(B, T, len(self.experts), device=x.device)
        one_hot.scatter_(-1, indices, 1.0)
        tokens_per_expert = one_hot.sum(dim=(0, 1)) / num_tokens  # (num_experts,)
        full_probs = F.softmax(expert_selections, dim=-1)
        mean_prob_per_expert = full_probs.mean(dim=(0, 1))  # (num_experts,)
        self.aux_loss = (tokens_per_expert * mean_prob_per_expert).sum() * len(self.experts)

        # create output matrix by selecting and adding expert results
        combined_output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # Find tokens assigned to this specific expert
            batch_coords, time_coords, topk_coords = torch.where(indices == i)
            expert_input = x[batch_coords, time_coords]
            expert_out = expert(expert_input)
            combined_output[batch_coords, time_coords] += (weights[batch_coords, time_coords, topk_coords].unsqueeze(-1) * expert_out)

        return combined_output