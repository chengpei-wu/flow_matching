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
    def __init__(self, input_dim, time_embed_dim=64):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(time_embed_dim)

        self.input_dim = input_dim
        self.latent_dim = input_dim + time_embed_dim  # full input = x_t + t_embed

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, input_dim),
        )

    def forward(self, x_t, t):
        t_embed = self.time_embed(t)  # shape: (B, time_embed_dim)
        xt = torch.cat([x_t, t_embed], dim=-1)  # concat (B, input_dim + time_embed_dim)
        return self.model(xt)

# class FlowMatchingMLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim=256, time_embed_dim=64, depth=4):
#         super().__init__()
#         self.time_embed = FourierTimeEmbedding(time_embed_dim)
#
#         self.input_layer = nn.Linear(input_dim + time_embed_dim, hidden_dim)
#
#         layers = []
#         for _ in range(depth - 2):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())
#         self.hidden_layers = nn.Sequential(*layers)
#
#         self.output_layer = nn.Linear(hidden_dim, input_dim)
#
#     def forward(self, x_t, t):
#         t_embed = self.time_embed(t)
#         xt = torch.cat([x_t, t_embed], dim=-1)
#         h = self.input_layer(xt)
#         h = self.hidden_layers(h)
#
#         return self.output_layer(h)

class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=16, downsample=False):
        super(DownLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.fc = nn.Linear(time_emb_dim, in_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool2d(2)

    def forward(self, x, temb):
        res = x
        x += self.fc(temb)[:, :, None, None]
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        res = self.shortcut(res) if self.shortcut is not None else res
        x = x + res
        if self.downsample:
            x = self.pool(x)
        return x

class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=16, upsample=False):
        super(UpLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.fc = nn.Linear(time_emb_dim, in_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.upsample = nn.Upsample(scale_factor=2) if upsample else None

    def forward(self, x, temb):
        if self.upsample:
            x = self.upsample(x)
        res = x
        x += self.fc(temb)[:, :, None, None]
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        res = self.shortcut(res) if self.shortcut is not None else res
        x = x + res
        return x

class MiddleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=16):
        super(MiddleLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.fc = nn.Linear(time_emb_dim, in_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x, temb):
        res = x
        x += self.fc(temb)[:, :, None, None]
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        res = self.shortcut(res) if self.shortcut is not None else res
        x = x + res
        return x

class MiniUnet(nn.Module):
    def __init__(self, base_channels=16, time_emb_dim=None):
        super(MiniUnet, self).__init__()
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim if time_emb_dim is not None else base_channels

        self.conv_in = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)
        self.down1 = nn.ModuleList([
            DownLayer(base_channels, base_channels * 2, self.time_emb_dim),
            DownLayer(base_channels * 2, base_channels * 2, self.time_emb_dim)
        ])
        self.maxpool1 = nn.MaxPool2d(2)
        self.down2 = nn.ModuleList([
            DownLayer(base_channels * 2, base_channels * 4, self.time_emb_dim),
            DownLayer(base_channels * 4, base_channels * 4, self.time_emb_dim)
        ])
        self.maxpool2 = nn.MaxPool2d(2)

        self.middle = MiddleLayer(base_channels * 4, base_channels * 4, self.time_emb_dim)

        self.upsample1 = nn.Upsample(scale_factor=2)
        self.up1 = nn.ModuleList([
            UpLayer(base_channels * 8, base_channels * 2, self.time_emb_dim),
            UpLayer(base_channels * 2, base_channels * 2, self.time_emb_dim)
        ])
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.up2 = nn.ModuleList([
            UpLayer(base_channels * 4, base_channels, self.time_emb_dim),
            UpLayer(base_channels, base_channels, self.time_emb_dim)
        ])

        self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=1)

    def time_emb(self, t, dim):
        t = t * 1000
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)

    def forward(self, x, t):
        x = self.conv_in(x)
        temb = self.time_emb(t, self.base_channels)
        for layer in self.down1:
            x = layer(x, temb)
        x1 = x
        x = self.maxpool1(x)
        for layer in self.down2:
            x = layer(x, temb)
        x2 = x
        x = self.maxpool2(x)

        x = self.middle(x, temb)

        x = torch.cat([self.upsample1(x), x2], dim=1)
        for layer in self.up1:
            x = layer(x, temb)
        x = torch.cat([self.upsample2(x), x1], dim=1)
        for layer in self.up2:
            x = layer(x, temb)

        x = self.conv_out(x)
        return x