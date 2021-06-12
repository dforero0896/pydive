from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import os
os.environ["CC"]="/home/astro/zhaoc/local/bin/gcc"
os.environ["CXX"]="/home/astro/zhaoc/local/bin/g++"
print(os.environ['LD_LIBRARY_PATH'])
print(os.environ['PATH'])


extra_compile_args=['-fPIC']
extra_link_args=[]
OMP=True
if OMP:
    extra_compile_args+=['-fopenmp']
    extra_link_args+=["-fopenmp"]


myext = Extension("pydive.pydive",
                  sources=['pydive/pydive.pyx',
                            #'pydive/delaunay_backend.cpp'
                            ],
                  include_dirs=[numpy.get_include(), 
                                '/home/astro/dforero/lib/CGAL-5.2.2/include', 
                                '/home/astro/zhaoc/local/lib/gcc/x86_64-pc-linux-gnu/10.3.0/include',
                                '/home/astro/zhaoc/local/lib/gcc/x86_64-pc-linux-gnu/10.3.0/include-fixed'
                                ],
                                
                  library_dirs=[
                                '/home/astro/dforero/lib/CGAL-5.2.2/build/lib',
                                '/home/astro/zhaoc/local/lib',
                                '/home/astro/zhaoc/local/lib64'
                                ],
                  libraries=['m', 'gsl', 'gslcblas', 'CGAL', 'gmp', 'mpfr'],
                  language='c++',
                  extra_compile_args=extra_compile_args,
                  extra_link_args=extra_link_args
             )

setup(name='pydive',
    author="Daniel Forero & Cheng Zhao",
    packages=find_packages(),
    ext_modules=cythonize([myext]))
