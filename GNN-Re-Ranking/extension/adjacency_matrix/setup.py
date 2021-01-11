from setuptools import setup, Extension

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='gnn_propagate',
    ext_modules=[
        CUDAExtension('gnn_propagate', [
            'gnn_propagate.cpp',
            'gnn_propagate_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext':BuildExtension  
    })