"""Fixed-step, trajectory-supervised alternatives to the baseline flow model.

Run with ``python -m src.stepwise`` from the repository root.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from src.model import MiniUnet
from src.utils.data_loader import get_dataloader


class TransportBlock(nn.Module):
    """Small convolutional map with no skip connection inside the block."""

    def __init__(self, channels, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Conv2d(width, channels, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class StepwiseTransport(nn.Module):
    """One separate block per fixed step; mode controls the update rule."""

    def __init__(self, channels, steps=20, width=16, mode="velocity_rec"):
        super().__init__()
        if steps < 1 or width < 4 or width % 4:
            raise ValueError("steps must be positive and width a positive multiple of 4")
        if mode not in ("velocity_rec", "direct"):
            raise ValueError(f"Unknown mode: {mode}")
        self.steps = steps
        self.mode = mode
        self.blocks = nn.ModuleList([TransportBlock(channels, width) for _ in range(steps)])
        if mode == "velocity_rec":
            # Initial rollout is the identity; the loss learns each displacement.
            for block in self.blocks:
                nn.init.zeros_(block.net[-1].weight)
                nn.init.zeros_(block.net[-1].bias)

    def step(self, x, index):
        prediction = self.blocks[index](x)
        if self.mode == "velocity_rec":
            return x + prediction / self.steps
        return prediction

    def forward(self, noise, target=None, return_trajectory=False):
        """Unroll using predicted states. Supervise every state if target is given.

        target has the shape of noise; the reference path is linear from the
        *same* input noise to target. No teacher forcing or velocity loss is used.
        """
        state = noise
        trajectory = [state] if return_trajectory else None
        loss = None
        for index in range(self.steps):
            state = self.step(state, index)
            if target is not None:
                fraction = (index + 1) / self.steps
                reference = (1 - fraction) * noise + fraction * target
                step_loss = F.mse_loss(state, reference)
                loss = step_loss if loss is None else loss + step_loss
            if return_trajectory:
                trajectory.append(state)
        if loss is not None:
            loss = loss / self.steps
        return state, loss, trajectory


@torch.no_grad()
def teacher_endpoint(teacher, noise, steps):
    """Pair each noise sample with its own pretrained FM teacher output."""
    state = noise
    for index in range(steps):
        t = noise.new_full((noise.shape[0],), index / steps)
        state = state + teacher(state, t) / steps
    return state


def build_teacher(checkpoint, channels, device):
    teacher = MiniUnet(num_channels=channels).to(device)
    teacher.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def train(model, dataloader, args, device, teacher):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        total, batches = 0.0, 0
        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images = images.to(device)
            noise = torch.randn_like(images)
            # Independent targets reproduce the original random FM pairing,
            # but are unsuitable for identifying a deterministic rollout.
            target = teacher_endpoint(teacher, noise, args.teacher_steps) if teacher else images
            _, loss, _ = model(noise, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        print(f"epoch={epoch + 1} mean_step_mse={total / batches:.6f}")


@torch.no_grad()
def sample(model, shape, device):
    model.eval()
    noise = torch.randn((1, *shape), device=device)
    _, _, trajectory = model(noise, return_trajectory=True)
    return [state[0].detach().cpu() for state in trajectory]


def save_trajectory(trajectory, destination):
    # Training data is normalized to [-1, 1].
    frames = [((image.clamp(-1, 1) + 1) / 2) for image in trajectory]
    fig, axes = plt.subplots(1, min(6, len(frames)), figsize=(12, 2))
    if not isinstance(axes, (list, tuple)):
        axes = list(axes.flat)
    if len(axes) > 1:
        indices = [round(i * (len(frames) - 1) / (len(axes) - 1))
                   for i in range(len(axes))]
    else:
        indices = [0]
    for ax, index in zip(axes, indices):
        image = frames[index]
        ax.imshow(image.permute(1, 2, 0) if image.shape[0] == 3 else image[0],
                  cmap=None if image.shape[0] == 3 else "gray", vmin=0, vmax=1)
        ax.set_title(f"step {index}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("velocity_rec", "direct"))
    parser.add_argument("--dataset", default="single_mnist-8")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--coupling", choices=("teacher", "independent"), default="teacher")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--teacher-steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.teacher_steps < 1:
        raise ValueError("teacher-steps must be positive")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(args.dataset, args.batch_size)
    images, _ = next(iter(dataloader))
    shape = images.shape[1:]
    checkpoint = args.checkpoint or Path(f"{args.dataset}_{args.mode}_{args.steps}step.pth")
    model = StepwiseTransport(shape[0], args.steps, args.width, args.mode).to(device)

    if args.sample:
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        expected = {"mode": args.mode, "steps": args.steps, "width": args.width,
                    "shape": tuple(shape)}
        for key, value in expected.items():
            if saved[key] != value:
                raise ValueError(f"checkpoint {key}={saved[key]} differs from requested {value}")
        model.load_state_dict(saved["state_dict"])
        output = args.output or Path(f"{args.dataset}_{args.mode}_trajectory.png")
        save_trajectory(sample(model, shape, device), output)
        print(f"Saved {output}")
        return

    teacher = None
    if args.coupling == "teacher":
        source = args.teacher_checkpoint or Path(f"{args.dataset}_flow_matching.pth")
        teacher = build_teacher(source, shape[0], device)
    train(model, dataloader, args, device, teacher)
    torch.save({"state_dict": model.state_dict(), "mode": args.mode,
                "steps": args.steps, "width": args.width, "shape": tuple(shape),
                "coupling": args.coupling}, checkpoint)
    print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
