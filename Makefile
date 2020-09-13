all: 
	python setup.py build_ext --inplace
	mv helpers* bin
clean:
	rm -f bin/helpers*