"""
Builds the deformable extension.
"""
import glob
import os

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME, CUDAExtension
from setuptools import find_packages, setup


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(this_dir, 'src')
    sources = [os.path.join(src_dir, 'deformable.cpp')]

    extension = CppExtension
    sources += glob.glob(os.path.join(src_dir, 'cpu', '*.cpp'))
    define_macros = []
    extra_compile_args = {'cxx': []}

    extension = CUDAExtension
    sources += glob.glob(os.path.join(src_dir, 'cuda', '*.cu'))
    define_macros += [('WITH_CUDA', None)]
    extra_compile_args['nvcc'] = [
        '-DCUDA_HAS_FP16=1',
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]

    ext_kwargs = {'include_dirs': [src_dir], 'define_macros': define_macros, 'extra_compile_args': extra_compile_args}
    ext_modules = [extension('deformable', sources, **ext_kwargs)]

    return ext_modules


setup(
    name="Deformable",
    version="0.1",
    author="Cédric Picron",
    description="Implementation of CUDA deformable functions.",
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.5.1",
        "torchvision>=0.6.1",
    ],
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
