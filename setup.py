from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import os
import subprocess
import sys

print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH', 'not set'))
print("PATH:", os.environ.get('PATH', 'not set'))

def check_cgal_via_pkgconfig():
    """Check if CGAL is available via pkg-config"""
    try:
        result = subprocess.run(['pkg-config', '--modversion', 'cgal'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Found CGAL via pkg-config: {result.stdout.strip()}")
            # Get compiler flags
            cflags = subprocess.run(['pkg-config', '--cflags', 'cgal'], 
                                  capture_output=True, text=True).stdout.strip().split()
            libs = subprocess.run(['pkg-config', '--libs', 'cgal'], 
                                capture_output=True, text=True).stdout.strip().split()
            return True, cflags, libs
    except Exception as e:
        print(f"pkg-config check failed: {e}")
    return False, [], []

def check_cgal_via_cmake():
    """Try building CGAL backend with CMake"""
    try:
        print("Attempting to build CGAL backend with CMake...")
        result = os.system("bash run_cmake.sh 2>&1 | tee /tmp/cgal_build.log")
        if result == 0 and os.path.exists('pydive/pydive.cpp'):
            print("CGAL backend built successfully with CMake")
            return True
        else:
            print("CMake build failed, check /tmp/cgal_build.log")
            return False
    except Exception as e:
        print(f"CMake build failed: {e}")
        return False

# Try different methods to detect/build CGAL
cgal_available = False
build_method = None
cgal_cflags = []
cgal_libs = []

# Method 1: Try pkg-config first (most portable)
cgal_pkgconfig, cgal_cflags, cgal_libs = check_cgal_via_pkgconfig()
if cgal_pkgconfig:
    cgal_available = True
    build_method = 'pkgconfig'
    print("Using CGAL detected via pkg-config")

# Method 2: Fall back to CMake build
if not cgal_available:
    if check_cgal_via_cmake():
        cgal_available = True
        build_method = 'cmake'
        print("Using CGAL backend built with CMake")

if not cgal_available:
    print("\n" + "="*70)
    print("WARNING: CGAL backend not available!")
    print("="*70)
    print("\nCGAL could not be found or built. You have two options:")
    print("\n1. Install CGAL and rebuild:")
    print("   - Conda: conda install -c conda-forge cgal boost-cpp gmp mpfr")
    print("   - Ubuntu: sudo apt-get install libcgal-dev libboost-all-dev")
    print("   - macOS: brew install cgal boost gmp mpfr")
    print("\n2. Use scipy-only mode (slower but easier to install)")
    print("   Note: A scipy backend is planned for future releases")
    print("="*70 + "\n")

extra_compile_args = ['-fPIC', '-std=c++11']
extra_link_args = []

# Add OpenMP if available
try:
    import multiprocessing
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')
except:
    pass

# Add CGAL-specific flags if using pkg-config
if build_method == 'pkgconfig':
    extra_compile_args.extend(cgal_cflags)
    extra_link_args.extend(cgal_libs)

ext_modules = []

# Build extension based on method
if cgal_available:
    if build_method == 'cmake' and os.path.exists('pydive/pydive.cpp'):
        # Use original CMake-generated pydive.cpp
        myext = Extension("pydive.pydive",
                          sources=['pydive/pydive.cpp'],
                          include_dirs=[numpy.get_include()],
                          library_dirs=[],
                          libraries=['m', 'gsl', 'gslcblas', 'gmp', 'mpfr'],
                          language='c++',
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args
                         )
        ext_modules.append(myext)
        print("Building extension from CMake-generated pydive.cpp")
    
    elif build_method == 'pkgconfig' or os.path.exists('pydive/delaunay_backend_standalone.cpp'):
        # Use new standalone backend with Cython
        myext = Extension("pydive.pydive",
                          sources=['pydive/delaunay_backend.pyx', 'pydive/delaunay_backend_standalone.cpp'],
                          include_dirs=[numpy.get_include(), '.'],
                          library_dirs=[],
                          libraries=['m', 'gmp', 'mpfr'],
                          language='c++',
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args
                         )
        ext_modules.append(myext)
        print("Building extension from standalone backend (delaunay_backend.pyx)")
else:
    print("Skipping CGAL extension build - will use scipy backend only")

setup(
    name='pydive',
    version='2.0.0',
    author="Daniel Forero",
    description="Python Delaunay triangulation-based void finder",
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'cython'],
    extras_require={
        'cgal': ['cython', 'numpy', 'cgal'],  # CGAL installation varies by platform
        'scipy': ['scipy', 'numpy']
    },
    ext_modules=cythonize(ext_modules, language_level=3) if ext_modules else [],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
)
