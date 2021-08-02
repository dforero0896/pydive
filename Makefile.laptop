all: 
	python setup.py build_ext --inplace
delaunay:
	g++ pydive/delaunay_backend.cpp -I/home/daniel/libraries/cgal/CGAL-5.1/include -I/home/daniel/anaconda3/envs/dive/include -I/home/daniel/anaconda3/envs/dive/include/boost -L/home/daniel/anaconda3/envs/dive/lib -L/home/daniel/libraries/cgal/CGAL-5.1/lib -lCGAL -lgmp -lmpfr -frounding-math -o delaunay_backend.o 
clean:
	rm -f pydive/*so pydive/helpers.c
