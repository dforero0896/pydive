from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

extra_compile_args=['-fPIC']
extra_link_args=[]
OMP=True
if OMP:
    extra_compile_args+=['-fopenmp']
    extra_link_args+=["-fopenmp"]


myext = Extension("pydive.pydive",
                  sources=['pydive/pydive.pyx'],
                  include_dirs=[numpy.get_include(), '/usr/include', '/global/common/sw/cray/cnl7/haswell/gsl/2.5/intel/19.0.3.199/7twqxxq/include'],
                  library_dirs=['/usr/lib/x86_64-linux-gnu', '/global/common/sw/cray/cnl7/haswell/gsl/2.5/intel/19.0.3.199/7twqxxq/lib'],
                  libraries=['m', 'gsl', 'gslcblas'],
                  language='c',
                  extra_compile_args=extra_compile_args,
                  extra_link_args=extra_link_args
             )

setup(name='pydive',
    author="Daniel Forero & Cheng Zhao",
    packages=find_packages(),
    ext_modules=cythonize([myext]))
