from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import shutil
import os
setup(
    name='mscclpp_ear',
    ext_modules=[
        CUDAExtension('mscclpp_ear', ['ear_engine.cu'],
        include_dirs=['/usr/local/mscclpp/include', "/opt/ompi/include"],
        library_dirs=['/usr/local/mscclpp/lib', "/opt/ompi/lib", "/usr/local/lib"],
        libraries=['mscclpp', 'mpi'],
        extra_cuda_cflags=['-arch=gfx942'],
        extra_hip_cflags=['-arch=gfx942'],
        extra_compile_args=['--offload-arch=gfx942', '-U__HIP_NO_HALF_CONVERSIONS__', '-U__HIP_NO_HALF_OPERATORS__'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

root_dir = os.path.dirname(os.path.abspath(__file__))

shutil.rmtree(os.path.join(root_dir, 'dist'))
shutil.rmtree(os.path.join(root_dir, 'mscclpp_ear.egg-info'))

for file in os.listdir(root_dir):
    if file.endswith('.hip'):
        os.remove(os.path.join(root_dir, file))
