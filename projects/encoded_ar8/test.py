import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import mscclpp_ear


def init_process(rank: int, world_size: int, master_addr: str) -> None:
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def test_ear(rank: int, world_size: int, m: int, n: int):
    try:
        # Setup
        init_process(rank, world_size, master_addr='127.0.0.1')
        x = torch.ones(size=(m, n), device=f"cuda:{rank}", dtype=torch.float16)
        # Run
        ear_engine = mscclpp_ear.EarEngine(rank)
        ear_engine.launchEncodedCrossReduce(x)
        # Check results
        expected = torch.full_like(x, world_size)
        assert torch.allclose(x, expected), f"Rank {rank}: Expected {expected}, got {x}"
    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise e


if __name__ == "__main__":
    m = 1
    n = 16384

    # Use all available GPUs
    world_size = torch.cuda.device_count()
    if world_size != 8:
        raise RuntimeError(f"Test requires 8 GPUs, but got {world_size = }")

    # Start multiple processes
    mp.spawn(
        test_ear,
        args=(world_size, m, n),
        nprocs=world_size,
        join=True,
        start_method='spawn'
    )
