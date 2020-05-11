from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='stensor', 
    ext_modules=[cpp_extension.CppExtension('stensor', ['stensor.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension})

