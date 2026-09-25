#export CC=/home/astro/zhaoc/local/bin/gcc
#export LD_LIBRARY_PATH=/home/astro/dforero/lib/CGAL-5.2.2/build/lib:/home/astro/zhaoc/local/lib:/home/astro/zhaoc/local/lib64:$LD_LIBRARY_PATH
#export PATH=/home/astro/zhaoc/local/bin:$PATH
INCLUDE_DIRS=-I/home/astro/dforero/lib/CGAL-5.2.2/include -I/home/astro/zhaoc/local/lib/gcc/x86_64-pc-linux-gnu/10.3.0/include  -I/home/astro/zhaoc/local/lib/gcc/x86_64-pc-linux-gnu/10.3.0/include-fixed
LIB_DIRS=-L/home/astro/dforero/lib/CGAL-5.2.2/build/lib #-L/home/astro/zhaoc/local/lib 


all: 
	python setup.py build_ext --inplace
delaunay:
	$(CXX) pydive/delaunay_backend.cpp $(INCLUDE_DIRS)  $(LIB_DIRS)  -lgmp -lmpfr -frounding-math -o delaunay_backend.o 
clean:
	rm -f pydive/*so pydive/pydive.cpp delaunay_backend.o pydive/delaunay_backend
	
