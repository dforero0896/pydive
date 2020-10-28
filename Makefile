all: 
	python setup.py build_ext --inplace
clean:
	rm -f pydive/*so pydive/helpers.c
