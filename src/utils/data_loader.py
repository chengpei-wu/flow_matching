import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA, MNIST


def get_dataloader(dataset: str, batch_size: 16) -> DataLoader:
    transform1 = transforms.Compose(
        [
            transforms.CenterCrop(168),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    transform2 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    if dataset == 'celeba':
        dataset = CelebA(root=f'dataset/', download=False, transform=transform1)
    elif dataset == 'mnist':
        dataset = MNIST(root=f'dataset/{dataset}/', download=True, transform=transform2)
    elif 'single_mnist' in dataset:
        cls = int(dataset.split('-')[1])
        dataset = MNIST(root=f'dataset/mnist/', download=True, transform=transform2)

        targets = dataset.targets.clone()
        mask = targets == cls

        dataset.data = dataset.data[mask]
        dataset.targets = targets[mask]
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    return DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, drop_last=True)


if __name__ == "__main__":
    mnist_loader = get_dataloader('celeba', batch_size=1)
    img = next(iter(mnist_loader))[0]
    # print(next(iter(mnist_loader)))
    img = img.squeeze()

    if img.dim() == 3:
        img = img.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.show()
    # img.save("tmp.jpg")
