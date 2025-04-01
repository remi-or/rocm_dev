import torch
import torch.distributed as dist

import os
import mscclpp_ear


class EncodedAllReduce:
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.init_process()
        self._ear_engine = mscclpp_ear.EarEngine(self.rank)

    def init_process(self, master_addr: str = "127.0.0.1") -> None:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.rank)

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        try:
            self._ear_engine.launchEncodedCrossReduce(x)
        except Exception as e:
            print(f"Error on rank {self.rank}: {str(e)}")
            raise e
        return x
