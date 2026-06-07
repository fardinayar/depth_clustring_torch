"""Optional ahead-of-time build:  python setup.py install
(The package also JIT-compiles the kernel on first use, so this is optional.)"""
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

setup(
    name="depth_clustering_torch",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="depth_clustering_torch._cuda",
            sources=["depth_clustering_torch/csrc/cc_cuda.cu"],
            extra_compile_args={"cxx": ["-O3"],
                                "nvcc": ["-O3", "--use_fast_math"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch>=1.12"],
)
