import torch
import torch.nn as nn
from tqdm import tqdm

from src.model import FlowMatchingMLP
from src.utils.args_parser import parse_args
from src.utils.data_loader import get_dataloader


def alpha_t(t):
    return 1 - t


def beta_t(t):
    return t


def x_t(z, epsilon, t):
    return alpha_t(t) * z + beta_t(t) * epsilon


def train(model, dataloader, device, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    loss_fn = nn.MSELoss()

    dataset_len = len(dataloader.dataset)

    for _ in tqdm(range(args.epochs)):
        loss_sum = 0
        for z, _ in dataloader:
            z = z.to(device).view(args.batch_size, -1)

            # sample gaussian noise
            epsilon = torch.randn_like(z).to(device)

            for step in range(args.num_time_steps):
                # sample time t uniformly form [0, 1]
                t = torch.rand(args.batch_size, 1)

                # sample x_t form p_t(x|z)
                xt = x_t(z, epsilon, t)

                # computing ground truth conditional vector field u_t(x|z) = \epsilon - z
                u_t = epsilon - z

                # computing predicted conditional vector field u^\theta_t(x|z) = neural_network(x_t, t)
                u_t_pred = model(xt, t)

                # computing MSE loss
                loss = loss_fn(u_t_pred, u_t)

                # backward propagation to update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss
        loss_sum /= dataset_len


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # init dataloader
    data_loader = get_dataloader(dataset=args.dataset, batch_size=args.batch_size)
    img_dim = next(iter(data_loader))[0].reshape(args.batch_size, -1).shape[1]

    # init neural network model
    model = FlowMatchingMLP(
        input_dim=img_dim
    )

    # train

    train(model, data_loader, device, args)
