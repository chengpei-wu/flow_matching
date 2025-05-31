import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.model import MiniUnet
from src.utils.args_parser import parse_args
from src.utils.data_loader import get_dataloader


def alpha_t(t):
    return t


def beta_t(t):
    return 1 - t


def x_t(z, epsilon, t):
    t = t[:, None, None, None]
    # t = t.expand(-1, img_shape[1], -1, -1)
    return alpha_t(t) * z + beta_t(t) * epsilon


def train(model, dataloader, device, args):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    with tqdm(desc=f'Training: ', total=args.epochs) as pbar:
        for epoch in range(args.epochs):
            all_loss = []
            for z, _ in dataloader:
                z = z.to(device)

                # sample gaussian noise
                epsilon = torch.randn_like(z).to(device)

                # sample time t uniformly form [0, 1]
                t = torch.rand(args.batch_size).to(device)

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


def sample(model, num_time_steps=50):
    model.eval()
    model = model.to('cpu')
    sinle_img_shape = (1, img_shape[1], img_shape[2], img_shape[3])
    epsilon = torch.randn(sinle_img_shape)
    h = 1 / num_time_steps
    ts = torch.arange(0, 1, h)

    trajectory = []

    with torch.no_grad():
        xt = epsilon
        trajectory.append(xt.clone().detach())
        for t in ts:
            t = t.reshape(1)
            xt = xt + h * model(xt, t)
            trajectory.append(xt.clone().detach())
    z = xt
    z = z.reshape(sinle_img_shape).squeeze().clamp(-1,1)
    return z.cpu(), trajectory


def visualize_trajectory(trajectory):
    # trajectory 是一个 list，每个元素形如 [1, C, H, W]
    images = [x.squeeze().detach().cpu() for x in trajectory]

    # 判断是灰度图还是 RGB 图
    is_rgb = images[0].dim() == 3 and images[0].shape[0] == 3  # [3, H, W]

    # 创建动画
    fig = plt.figure(figsize=(4, 4))

    if is_rgb:
        def to_numpy(img):
            return img.permute(1, 2, 0).numpy().clip(0, 1)
    else:
        def to_numpy(img):
            return img.squeeze().numpy().clip(0, 1)

    im = plt.imshow(to_numpy(images[0]), cmap='gray' if not is_rgb else None)

    def update(frame):
        im.set_data(to_numpy(images[frame]))
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(images), interval=50, blit=True)
    plt.axis('off')
    plt.title("Flow Matching Sampling Trajectory")
    # plt.show()

    # 如果想保存为 gif
    ani.save("trajectory.gif", writer='pillow')


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # init dataloader
    data_loader = get_dataloader(dataset=args.dataset, batch_size=args.batch_size)
    img = next(iter(data_loader))[0]
    img_shape = img.shape
    img_dim = img.shape[1] * img.shape[2] * img.shape[3]  # assuming img is in shape (B, C, H, W)

    # init neural network model
    model = MiniUnet(
        num_channels=img.shape[1]
    )

    if not args.sample:
        # train the model
        train(model, data_loader, device, args)

        # save the model
        torch.save(model.state_dict(), f'{args.dataset}_flow_matching.pth')
        print(f'Model saved {args.dataset}_flow_matching.pth')
    else:
        # load the model
        model.load_state_dict(torch.load(f'{args.dataset}_flow_matching.pth', weights_only=True))
        print(f'Model loaded from {args.dataset}_flow_matching.pth')

        # sample from the model
        img, tra = sample(model)

        visualize_trajectory(tra)
        if img.dim() == 3:
            img = img.permute(1, 2, 0).cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f'Sampled Image from {args.dataset}')
        plt.show()
