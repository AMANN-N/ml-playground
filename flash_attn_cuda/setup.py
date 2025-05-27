from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attn_cuda',
    ext_modules=[
        CUDAExtension(
            name='flash_attn_cuda',
            sources=['flash_attn.cpp', 'flash_attn.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
