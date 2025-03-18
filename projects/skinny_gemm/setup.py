from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, _find_rocm_home, IS_HIP_EXTENSION, CUDAExtension

print(f"ISHIP: {IS_HIP_EXTENSION}")

setup(
    name="hip_fused_gemm_nccl",
    ext_modules=[
        CUDAExtension(
            name="hip_fused_gemm_nccl",
            sources=["skinny_gemm.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
