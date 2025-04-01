import torch
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from ear_class import EncodedAllReduce

STEP = 16


def generate_input(rank: int, m: int, n: int):
    torch.manual_seed(rank)
    return torch.randn(size=(m, n), device="cuda", dtype=torch.float16)


def test_ear(rank: int, world_size: int, m: int, n: int):
    # Setup
    ear = EncodedAllReduce(world_size, rank)
    x = generate_input(rank, m, n)
    # Run
    ear.reduce(x)

    # Display first line of all ranks
    print(f"Rank {rank} : {x[0, :STEP].tolist()}")

    # Display results for rank 0
    # if rank == 0:
    #     for i in range(0, n, STEP):
    #         print(f"{i}: ", end=" | ")
    #         for j in range(STEP):
    #             print(f"{x[0, i + j].item():.2f}", end=" | ")
    #         print()

    # Compute expected result
    excpected = torch.zeros_like(x)
    for i in range(world_size):
        excpected += generate_input(i, m, n)
    if rank == 0:
        print(f"Expected : {excpected[0, :STEP].tolist()}")

    # Check result
    allclose = torch.allclose(x, excpected)
    if allclose:
        print(f"Rank {rank} : passed allclose")
    else:
        delta = x.sub(excpected).abs()
        print(f"Rank {rank} : failed allclose with {delta.max().item() = }")
        print(f"Rank {rank} : failed allclose with {delta.mean().item() = }")

    # Rank 0 saves a figure of result and expected
    if rank == 0:
        linewidth = 256
        fig, ax = plt.subplots(1, 3, figsize=(15, 3))
        ax[0].matshow(x.reshape(-1, linewidth).numpy(force=True))
        ax[1].matshow(excpected.reshape(-1, linewidth).numpy(force=True))
        ax[2].matshow(delta.reshape(-1, linewidth).numpy(force=True))
        cax = ax[2].matshow(delta.reshape(-1, linewidth).numpy(force=True))
        plt.colorbar(cax, ax=ax[2], orientation='horizontal', pad=0.01)
        plt.savefig(f"_ear_test.png")


if __name__ == "__main__":
    m = 2
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
