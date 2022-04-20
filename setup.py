from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import os
print(os.environ['LD_LIBRARY_PATH'])
print(os.environ['PATH'])
os.system("bash run_cmake.sh")

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
                                '/global/u1/d/dforero/lib/CGAL-5.4/include', 
				'/global/common/software/nersc/cori-2022q1/spack/cray-cnl7-haswell/gsl-2.7-ihnf7gi/include',
				"/global/homes/d/dforero/.conda/envs/jax/include"
                                ],
                                
                  library_dirs=[
                                '/home/astro/dforero/lib/CGAL-5.2.2/build/lib',
				"/global/common/software/nersc/cori-2022q1/spack/cray-cnl7-haswell/gsl-2.7-ihnf7gi/lib",
				"/global/homes/d/dforero/.conda/envs/jax/lib"
                                ],
                  libraries=['m', 'gsl', 'gslcblas', 'gmp', 'mpfr'],
                  language='c++',
                  extra_compile_args=extra_compile_args,
                  extra_link_args=extra_link_args
             )

setup(name='pydive',
    author="Daniel Forero",
    packages=find_packages(),
    ext_modules=cythonize([myext]))
