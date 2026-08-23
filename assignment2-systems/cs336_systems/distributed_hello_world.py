import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_worker(rank, world_size):
    setup(rank, world_size)
    # Your distributed code here
    data=torch.randint(0,10,(3,))
    print(f"Rank {rank} before broadcast: {data}")
    dist.broadcast(data, src=0)
    print(f"Rank {rank} after broadcast: {data}")

if __name__ == "__main__":
    world_size = 4  # Number of processes
    mp.spawn(distributed_worker, args=(world_size,), nprocs=world_size, join=True)