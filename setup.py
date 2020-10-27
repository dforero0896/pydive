from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
import numpy

myext = Extension("helpers",
                  sources=['src/helpers.pyx'],
                  include_dirs=[numpy.get_include(), '/usr/include'],
                  library_dirs=['/usr/lib/x86_64-linux-gnu'],
                  libraries=['m', 'gsl', 'gslcblas'],
                  language='c',
                 # extra_compile_args=["-std=c++11"],
                 # extra_link_args=["-std=c++11"]
             )

setup(name='helpers',
      ext_modules=cythonize([myext]))