from torchvision import datasets
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from time import perf_counter

def GetTrainTestLoaders(num_workers = 1, load_test = False):
    transformImages = Compose([
    RandomCrop(size=32, padding=[4]),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    train = datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=transformImages
    )
    test = datasets.CIFAR10(
        root="data",
        train=False,
        download=False,
        transform=transformImages
    )
    train_dataloader = DataLoader(train, batch_size=128, shuffle=True, num_workers=num_workers)
    if(load_test):
        test_dataloader = DataLoader(test, batch_size=100, shuffle=True, num_workers=num_workers)
        return train_dataloader, test_dataloader
    else:
        return train_dataloader, None

