from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, _find_rocm_home, IS_HIP_EXTENSION, CUDAExtension
import os
import shutil
import glob

os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"

print(f"ISHIP: {IS_HIP_EXTENSION}")

setup(
    name="hfrk_skinny_gemm",
    ext_modules=[
        CUDAExtension(
            name="hfrk_skinny_gemm",
            sources=["skinny_gemm_binding.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

root = os.path.dirname(os.path.abspath(__file__))

files_to_remove = glob.glob(os.path.join(root, "*.hip"))
files_to_remove += glob.glob(os.path.join(root, "*_hip.cuh"))
for file in files_to_remove:
    os.remove(file)

shutil.rmtree(os.path.join(root, "build"))
shutil.rmtree(os.path.join(root, "dist"))
shutil.rmtree(os.path.join(root, "hfrk_skinny_gemm.egg-info"))
