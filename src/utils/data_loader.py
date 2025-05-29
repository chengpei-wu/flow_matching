import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA, MNIST


def get_dataloader(dataset: str, batch_size: 16) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.CenterCrop(168),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )
    if dataset == 'celeba':
        dataset = CelebA(root=f'dataset/', download=False, transform=transform)
    elif dataset == 'mnist':
        dataset = MNIST(root=f'dataset/{dataset}/', download=False, transform=transforms.ToTensor())
    elif 'single_mnist' in dataset:
        cls = int(dataset.split('-')[1])
        print(cls)
        dataset = MNIST(root=f'dataset/mnist/', download=False, transform=transforms.ToTensor())

        targets = dataset.targets.clone()
        mask = targets == cls

        dataset.data = dataset.data[mask]
        dataset.targets = targets[mask]
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    return DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)


if __name__ == "__main__":
    mnist_loader = get_dataloader('single_mnist-4', batch_size=16)
    img = next(iter(mnist_loader))[0]

    print(img.shape)
    N, C, H, W = img.shape
    assert N == 16
    img = torch.permute(img, (1, 0, 2, 3))
    img = torch.reshape(img, (C, 4, 4 * H, W))
    img = torch.permute(img, (0, 2, 1, 3))
    img = torch.reshape(img, (C, 4 * H, 4 * W))
    img = transforms.ToPILImage()(img)
    img.show()
    # img.save("tmp.jpg")
