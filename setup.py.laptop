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
                  sources=['pydive/pydive.pyx',
                            #'pydive/delaunay_backend.cpp'
                            ],
                  include_dirs=[numpy.get_include(), '/usr/include', '/usr/local/inculde'
                                '/global/common/sw/cray/cnl7/haswell/gsl/2.5/intel/19.0.3.199/7twqxxq/include', 
                                #'/home/daniel/libraries/cgal/CGAL-5.1/include',
                                #'/home/daniel/libraries/CGAL-4.9/include',
                                '/home/daniel/anaconda3/envs/dive/include'
                                '/home/daniel/anaconda3/envs/dive/include/boost'],
                                
                  library_dirs=['/usr/local/lib',
                                '/usr/lib/x86_64-linux-gnu', 
                                #'/global/common/sw/cray/cnl7/haswell/gsl/2.5/intel/19.0.3.199/7twqxxq/lib',
                                '/home/daniel/anaconda3/envs/dive/lib',
                                #'/home/daniel/libraries/cgal/CGAL-5.1/lib'
                                #'/home/daniel/libraries/CGAL-4.9/lib'
                                
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
