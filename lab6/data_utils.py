from torchvision import datasets
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from time import perf_counter
from torch.utils.data.distributed import DistributedSampler

def GetTrainTestLoaders(num_workers = 1, load_test = False, batch_size=32, rank=None, world_size=None):
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
    train_sampler = None
    test_sampler = None
    if rank is not None and world_size is not None:
        train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank)
        if load_test:
            test_sampler = DistributedSampler(test, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=num_workers)
    
    if load_test:
        test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=(test_sampler is None), sampler=test_sampler, num_workers=num_workers)
        return train_dataloader, test_dataloader
    else:
        return train_dataloader, None