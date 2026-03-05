import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_size = int(4 * d_model * 2/3)  # hidden dim smaller than with ReLU because SwiGLU adds a linear layer (more params).
        self.fc1 = torch.nn.Linear(d_model, hidden_size)
        self.fc2 = torch.nn.Linear(d_model, hidden_size)
        self.silu = torch.nn.SiLU()
        self.fc3 = torch.nn.Linear(hidden_size, d_model)

    def forward(self, x):
        x_swiglu = self.fc1(x) * self.silu(self.fc2(x))
        out = self.fc3(x_swiglu)
        return out