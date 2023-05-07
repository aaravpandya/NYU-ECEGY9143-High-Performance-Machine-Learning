import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from train_utils import train
import argparse
from utils import write_to_file, init
import torch
import os
def main(args, rank=None, world_size=None):
    question = args.question
    output_file = args.output_file
    print(f'Running question {question}')
    if question == 'q1':
        d = {question: []}
        batch_size = 32
        epochs = 2
        print(f'{batch_size=}')
        while True:
            try:
                a = init(args, batch_size_arg=batch_size)
                if(a['device'] == None):
                    a['device'] = 'gpu'
                    model.to(a['device'])
                print(f"{a['device']=}")
                _, _, warmup_times = train(a['train_loader'], a['device'], a['optimizer'], a['model'], a['criterion'], 1)
                loss, acc, running_times = train(a['train_loader'], a['device'], a['optimizer'], a['model'], a['criterion'], epochs - 1)
                output = {'Top Accuracy': acc, 'Losses': loss, 'Warmup_times': warmup_times, 'Running_times': running_times, 'batch_size': batch_size}
                print(output)
                d[question].append(output)
                batch_size *= 4
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    break
                else:
                    raise e
        write_to_file(d, output_file)
        print(d)
    elif question == 'q2' or question == 'q3':
        d = {question: []}
        batch_size = 32
        epochs = 2
        print(f'{batch_size=} {rank=} {world_size=}')
        while True:
            try:
                a = init(args, rank=rank, world_size=world_size, batch_size_arg=batch_size)
                if(a['device'] == None):
                    a['device'] = f'cuda:{rank}'
                    a['model'].to(a['device'])
                print(f"{a['device']=}")
                _, _, warmup_times = train(a['train_loader'], a['device'], a['optimizer'], a['model'], a['criterion'], 2)
                loss, acc, running_times = train(a['train_loader'], a['device'], a['optimizer'], a['model'], a['criterion'], 1)
                output = {'Top Accuracy': acc, 'Losses': loss, 'Warmup_times': warmup_times, 'Running_times': running_times, 'batch_size': batch_size}
                print(output)
                d[question].append(output)
                batch_size *= 4
                if(batch_size > 512):
                    break
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    break
                else:
                    raise e
        output_file_with_rank = f"output_rank_{rank}.json"
        write_to_file(d, output_file_with_rank)
        print(d)


def run(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    main(args, rank, world_size)

    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output_file', type=str, default='output.json')
    parser.add_argument('--question', type=str, default='c2')
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    if(args.question=='q2'):
        world_size = 2
    if world_size > 1 and (args.question == 'q2' or args.question == 'q3'):
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main(args)
