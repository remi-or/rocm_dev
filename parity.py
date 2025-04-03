import argparse
import torch
import matplotlib

from class_project import Project

def extract_delta(chunk:str) -> float:
    return float(chunk.split(":")[1])

if __name__ == "__main__":

    # Retrieve the name of the project
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", "-p", required=True)
    parser.add_argument("--arguments", "-a", default="")
    args = parser.parse_args()
    project = Project(str(args.project))
    arguments = str(args.arguments)

    # Compile and run test
    test_output = project.compile_and_run("test.cu", arguments)
    deltas, infos = test_output.split(", {")
    print(infos)

    # Extract deltas
    deltas = deltas.split(", ")
    deltas = list(map(extract_delta, deltas))
    deltas = torch.tensor(deltas)

    m, n, k = list(map(int, arguments.split(" ")))
    deltas = deltas.reshape(m, n)

    matplotlib.image.imsave('skg_deltas.png', deltas.numpy(force=True))

    # fig, ax = plt.subplots()
    # cax = ax.pcolormesh(
    #     torch.arange(m).numpy(force=True),
    #     torch.arange(n).numpy(force=True),
    #     deltas.numpy(force=True),
    # )

    # # plt.colorbar(cax)
    # # ax.set_xlabel("m")
    # # ax.set_ylabel("n")
    # # ax.set_title('Deltas Heatmap')

    # # ax.set_xticks(list(range(0, n, 16)))

    # plt.savefig('deltas_heatmap.png')
    # # print(deltas)


"""
run with
python parity.py -p skinny_gemm -a "16 6720 2048
"""
