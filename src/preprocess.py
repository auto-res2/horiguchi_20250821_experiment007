import os
from torch.utils import data
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_loaders(name='MNIST', batch_size=128, aug=True, subset_train=None, subset_test=None):
    if name == 'MNIST':
        n_classes = 10
        ds_train = datasets.MNIST(root='./data', train=True, download=True)
        ds_test = datasets.MNIST(root='./data', train=False, download=True)
    elif name == 'FashionMNIST':
        n_classes = 10
        ds_train = datasets.FashionMNIST(root='./data', train=True, download=True)
        ds_test = datasets.FashionMNIST(root='./data', train=False, download=True)
    elif name == 'EMNIST':
        n_classes = 47
        ds_train = datasets.EMNIST(root='./data', split='balanced', train=True, download=True)
        ds_test = datasets.EMNIST(root='./data', split='balanced', train=False, download=True)
    else:
        raise ValueError(f'Unknown dataset: {name}')

    T_aug = []
    if aug:
        T_aug += [transforms.RandomAffine(degrees=5, translate=(0.07, 0.07))]
    T_aug += [transforms.ToTensor()]
    tf_train = transforms.Compose(T_aug)
    tf_test = transforms.ToTensor()

    ds_train.transform = tf_train
    ds_test.transform = tf_test

    if subset_train is not None:
        ds_train = Subset(ds_train, list(range(subset_train)))
    if subset_test is not None:
        ds_test = Subset(ds_test, list(range(subset_test)))

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return loader_train, loader_test, n_classes
