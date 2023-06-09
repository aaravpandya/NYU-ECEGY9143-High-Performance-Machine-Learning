import time
import torch
import torch.distributed as dist

def measure_latency_bandwidth(rank):
    # Number of iterations for averaging the measurements
    iterations = 100

    # Prepare small and large tensors for latency and bandwidth measurement
    small_tensor = torch.tensor([0.0], device=f'cuda:{rank}')
    large_tensor = torch.rand(10000000, device=f'cuda:{rank}')  # 1 million elements

    # Ensure synchronization before starting the timer
    torch.cuda.synchronize()

    # Measure latency
    latency_start_time = time.time()
    for _ in range(iterations):
        dist.broadcast(small_tensor, src=rank)
        torch.cuda.synchronize()
    latency_end_time = time.time()

    # Measure bandwidth
    bandwidth_start_time = time.time()
    for _ in range(iterations):
        dist.broadcast(large_tensor, src=rank)
        torch.cuda.synchronize()
    bandwidth_end_time = time.time()

    T_latency = (latency_end_time - latency_start_time) / iterations
    T_bandwidth = (large_tensor.numel() * 4) / ((bandwidth_end_time - bandwidth_start_time) / iterations)  # Assuming float32, which is 4 bytes per element
    T_bandwidth = T_bandwidth / (1024 * 1024 * 1024)
    return T_latency, T_bandwidth

