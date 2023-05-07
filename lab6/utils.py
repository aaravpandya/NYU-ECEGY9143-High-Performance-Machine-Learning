import os
import json
import torch
from train_utils import GetOptimizer, GetModel, GetDevice
from data_utils import GetTrainTestLoaders
from torch.nn.parallel import DistributedDataParallel as DDP

def write_to_file(data, filename):
    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        with open(filename, 'r+') as f:
            json_data = json.load(f)
            json_data.update(data)
            f.seek(0)
            f.truncate()
            json.dump(json_data, f, indent=4)


def find_best_num_workers(filename):
    data = {}
    with open(filename, 'r') as f:
        data = json.load(f)
    min_loading_time = float('inf')
    num_workers = None
    for item in data['c3']:
        loading_time = item['Running_times'][0]['Data loading time']
        if loading_time < min_loading_time:
            min_loading_time = loading_time
            num_workers = item['num_workers']
    return num_workers

def init(args, rank=None, world_size=None, workers=None, device_arg=None, optimizer_arg=None, batch_size_arg=32):
    preffered_device = device_arg if device_arg is not None else args.device
    num_workers = workers if workers is not None else args.num_workers
    optimizer = optimizer_arg if optimizer_arg is not None else args.optimizer
    epochs = args.epochs
    question = args.question
    output_file = args.output_file

    train_loader, _ = GetTrainTestLoaders(num_workers, batch_size=batch_size_arg, rank=rank, world_size=world_size)
    device = GetDevice(preffered_device, rank=rank)
    model = GetModel().to(device)  # Remove the `device` parameter
    if rank is not None and world_size is not None:
        model = DDP(model, device_ids=[rank], output_device=rank)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = GetOptimizer(model, opt=optimizer)
    d = {
    "num_workers": num_workers,
    "optimizer": optimizer,
    "epochs": epochs,
    "question": question,
    "output_file": output_file,
    "train_loader": train_loader,
    "device": device,
    "model": model,
    "criterion": criterion
    }
    return d