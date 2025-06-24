# data_loader/__init__.py

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataloader(cfg):
    """
    Returns (train_loader, val_loader, test_loader) based on cfg.
    Expects in cfg:
      - 'dataset': 'cifar10' or 'cifar100'
      - 'batch_size': int
      - 'data_dir': str
    """
    dataset_name = cfg.get("dataset", "cifar10").lower()
    batch_size   = cfg.get("batch_size", 128)
    data_dir     = cfg.get("data_dir", "./data")

    # normalization stats for CIFAR
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if dataset_name == "cifar10":
        full_train = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_set = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
    elif dataset_name == "cifar100":
        full_train = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_set = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # split train → train/val (90/10)
    n_train = int(0.9 * len(full_train))
    n_val   = len(full_train) - n_train
    train_set, val_set = random_split(full_train, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader
