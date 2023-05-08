import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from LatencyMeasure import measure_latency_bandwidth
import os
def init_process(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def worker(rank, world_size):
    init_process(rank, world_size)
    T_latency, T_bandwidth = measure_latency_bandwidth(rank)
    print(f"GPU {rank}: T_latency: {T_latency}, T_bandwidth: {T_bandwidth}")

def main():
    world_size = 4
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()

