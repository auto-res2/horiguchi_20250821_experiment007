from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist_loaders(train_subset: int = 1024, val_subset: int = 256, test_subset: int = 512, batch_size_train: int = 128, batch_size_eval: int = 256) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    full_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_idx = list(range(min(train_subset, len(full_train))))
    val_idx = list(range(min(val_subset, len(full_train))))
    test_idx = list(range(min(test_subset, len(full_test))))

    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=batch_size_train, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(Subset(full_train, val_idx), batch_size=batch_size_eval, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(Subset(full_test, test_idx), batch_size=batch_size_eval, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader


def get_small_mnist_loaders(subset: int = 512, batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader_small = DataLoader(Subset(train_ds, list(range(min(subset, len(train_ds))))), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader_small = DataLoader(Subset(test_ds, list(range(min(subset, len(test_ds))))), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    return train_loader_small, test_loader_small
