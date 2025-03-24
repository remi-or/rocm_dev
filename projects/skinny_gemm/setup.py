from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, _find_rocm_home, IS_HIP_EXTENSION, CUDAExtension
import os

os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"

print(f"ISHIP: {IS_HIP_EXTENSION}")

setup(
    name="hfrk_skinny_gemm",
    ext_modules=[
        CUDAExtension(
            name="hfrk_skinny_gemm",
            sources=["skinny_gemm.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
