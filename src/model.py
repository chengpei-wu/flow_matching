import math

import torch
import torch.nn as nn


class FourierTimeEmbedding(nn.Module):
    def __init__(self, embed_dim, max_freq=10.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.freqs = torch.exp(torch.linspace(0., math.log(max_freq), embed_dim // 2))

    def forward(self, t):
        # t: shape (B, 1)
        freqs = self.freqs.to(t.device)
        t_proj = t * freqs  # (B, D//2)
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class FlowMatchingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, time_embed_dim=64, depth=4):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(time_embed_dim)

        self.input_layer = nn.Linear(input_dim + time_embed_dim, hidden_dim)

        layers = []
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_t, t):
        t_embed = self.time_embed(t)
        xt = torch.cat([x_t, t_embed], dim=-1)
        h = self.input_layer(xt)
        h = self.hidden_layers(h)

        return self.output_layer(h)
