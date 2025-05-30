import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.model import FlowMatchingMLP
from src.utils.args_parser import parse_args
from src.utils.data_loader import get_dataloader


def alpha_t(t):
    return t


def beta_t(t):
    return 1 - t


def x_t(z, epsilon, t):
    return alpha_t(t) * z + beta_t(t) * epsilon


def train(model, dataloader, device, args):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    with tqdm(desc=f'Training: ', total=args.epochs) as pbar:
        for epoch in range(args.epochs):
            all_loss = []
            for z, _ in dataloader:
                z = z.to(device).reshape(args.batch_size, -1)

                # sample gaussian noise
                epsilon = torch.randn_like(z).to(device)

                # for step in range(args.num_time_steps):
                # sample time t uniformly form [0, 1]
                t = torch.rand(args.batch_size, 1).to(device)
                # t = torch.full((args.batch_size, 1), 0.5).to(device)

                # sample x_t form p_t(x|z)
                xt = x_t(z, epsilon, t)

                # computing ground truth conditional vector field u_t(x|z) = \epsilon - z
                u_t = z - epsilon

                # computing predicted conditional vector field u^\theta_t(x|z) = neural_network(x_t, t)
                u_t_pred = model(xt, t)

                # computing MSE loss
                loss = loss_fn(u_t_pred, u_t)

                # backward propagation to update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                all_loss.append(loss.item())

            pbar.set_postfix(
                {
                    'Epoch': epoch,
                    'Loss': np.mean(all_loss),
                }
            )
            pbar.update(1)


def sample(model, num_time_steps=1000):
    model.eval()
    model = model.to('cpu')

    epsilon = torch.randn(1, img_dim)
    h = 1 / num_time_steps
    print(h)
    ts = torch.arange(0, 1, h)

    with torch.no_grad():
        xt = epsilon
        for t in ts:
            t = t.view(1, 1)
            xt = xt + h * model(xt, t)

    z = xt
    z = z.reshape(img_shape).squeeze().clamp(min=-1, max=1)
    return z.cpu()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # init dataloader
    data_loader = get_dataloader(dataset=args.dataset, batch_size=args.batch_size)
    img = next(iter(data_loader))[0]
    img_shape = img.shape[1:]
    img_dim = img.shape[1] * img.shape[2] * img.shape[3]  # assuming img is in shape (B, C, H, W)

    # init neural network model
    model = FlowMatchingMLP(
        input_dim=img_dim,
    )
    if not args.sample:
        # train the model
        train(model, data_loader, device, args)

        torch.save(model.state_dict(), f'{args.dataset}_flow_matching.pth')
        print(f'Model saved {args.dataset}_flow_matching.pth')
    else:
        # load the model
        model.load_state_dict(torch.load(f'{args.dataset}_flow_matching.pth', weights_only=True))
        print(f'Model loaded from {args.dataset}_flow_matching.pth')

        # sample from the model
        img = sample(model)
        print(img.shape)
        plt.imshow(img, cmap='gray')
        plt.show()
