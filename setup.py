from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

extra_compile_args=['-fPIC']
extra_link_args=[]
OMP=True
if OMP:
    extra_compile_args+=['-fopenmp']
    extra_link_args+=["-fopenmp"]


myext = Extension("pydive.helpers",
                  sources=['pydive/helpers.pyx'],
                  include_dirs=[numpy.get_include(), '/usr/include', '/home/astro/dforero/lib/gsl-1.13/include','/home/astro/dforero/.conda/envs/mybase/include'],
                  library_dirs=['/usr/lib/x86_64-linux-gnu', '/home/astro/dforero/lib/gsl-1.13/lib', '/home/astro/dforero/.conda/envs/mybase/lib'],
                  libraries=['m', 'gsl', 'gslcblas'],
                  language='c',
                  extra_compile_args=extra_compile_args,
                  extra_link_args=extra_link_args
             )

setup(name='pydive',
    author="Daniel Forero & Cheng Zhao",
    packages=find_packages(),
    ext_modules=cythonize([myext]))
