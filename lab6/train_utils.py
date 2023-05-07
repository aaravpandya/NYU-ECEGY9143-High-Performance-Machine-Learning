from torch.utils.data import DataLoader
from torch.optim import Optimizer, SGD, Adam, Adagrad, Adadelta
from torch.nn import Module
from torchvision.models import resnet18
import torch
import numpy as np
from time import perf_counter

def train_one_epoch(train_dataloader: DataLoader, device, optimizer, model: Module, criterion) -> tuple:
    running_loss = 0
    correct = 0
    total = 0
    data_loading_time = 0
    training_time = 0
    communication_time = 0
    loops = 0
    start_time = perf_counter()
    data_loading_start_time = perf_counter()
    for _, data in enumerate(train_dataloader, 0):
        data_loading_time += perf_counter() - data_loading_start_time
        loops+=1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        training_start_time = perf_counter()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).float().sum().item()
        loss = criterion(outputs, labels)
        communication_start_time = perf_counter()
        loss.backward()
        optimizer.step()
        communication_time += perf_counter() - communication_start_time

        training_time += perf_counter() - training_start_time

        # print statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        data_loading_start_time = perf_counter()
        
    total_running_time_for_epoch = perf_counter() - start_time
    epoch_loss = running_loss / len(train_dataloader.dataset)
    epoch_acc = correct / total
    epoch_info = [epoch_loss, epoch_acc]
    running_times = {
        "Data loading time": data_loading_time,
        "Training time": training_time,
        "Communication time": communication_time,
        "Total epoch time": total_running_time_for_epoch
    }
    
    return epoch_info, running_times
def train(train_dataloader: DataLoader, device, optimizer, model: Module, criterion, epochs: int) -> tuple:
    losses = []
    accs = []
    accumulated_times = []
    for i in range(epochs):
        epoch_info, times = train_one_epoch(train_dataloader,device,optimizer,model,criterion)
        loss, acc = epoch_info
        losses.append(loss)
        accs.append(acc)
        accumulated_times.append(times)
    topacc = accs[np.argmax(accs)]
    return losses, topacc, accumulated_times

def GetOptimizer(model: Module,opt: str = 'sgd'):
    lr = 0.1
    momentum=0.9
    weight_decay=5e-4
    if(opt=='sgd'):
        return SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
    if(opt=='sgdN'):
        return SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay, nesterov=True)
    if(opt=='adam'):
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if(opt=='adadelta'):
        return Adadelta(model.parameters(),lr=lr,weight_decay=weight_decay)
    if(opt=='adagrad'):
        return Adagrad(model.parameters(),lr=lr,weight_decay=weight_decay)

def GetModel():
    model = resnet18()
    model.train()
    return model


def GetDevice(preffered_device, rank=None):
    if preffered_device == 'gpu' and torch.cuda.is_available():
        if rank is not None:
            return torch.device(f'cuda:{rank}')
        else:
            return torch.device('cuda:0')
    else:
        return torch.device('cpu')




