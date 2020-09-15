all: 
	python setup.py build_ext --inplace
	mv helpers* bin
	mv cgal_helpers* bin
clean:
	rm -f bin/helpers* bin/cgal_helpers*