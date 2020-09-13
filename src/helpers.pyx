import cython
from cython.parallel import prange
import numpy as np
from libc.stdlib cimport malloc, free
from libc.stdio cimport FILE, fprintf, fopen, fclose, printf



cdef extern from "math.h":
    double sqrt(double x) nogil
ctypedef int size_t
cdef extern from "gsl/gsl_block.h":
    ctypedef struct gsl_block:
        size_t size
        double * data
cdef extern from "gsl/gsl_permutation.h":
    ctypedef struct gsl_permutation:
        size_t size
        size_t * data
    gsl_permutation *  gsl_permutation_alloc(size_t n) nogil
cdef extern from "gsl/gsl_matrix.h":
    ctypedef struct gsl_matrix:
        size_t size1
        size_t size2
        size_t tda
        double * data
        gsl_block * block
        int owner
    ctypedef struct gsl_matrix_view:
        gsl_matrix matrix
    gsl_matrix_view_array (double * base,
                             const size_t n1, 
                             const size_t n2)
    gsl_matrix *  gsl_matrix_alloc(size_t n1, size_t n2) nogil
    double  gsl_matrix_get(gsl_matrix * m, size_t i, size_t j) nogil
    void  gsl_matrix_set(gsl_matrix * m, size_t i, size_t j, double x) nogil
    void  gsl_matrix_free(gsl_matrix * m) nogil
    gsl_matrix_view  gsl_matrix_submatrix(gsl_matrix * m, size_t k1, size_t k2, size_t n1, size_t n2) nogil
    int  gsl_matrix_memcpy(gsl_matrix * dest, gsl_matrix * src) nogil


cdef extern from "gsl/gsl_linalg.h":
    int gsl_linalg_LU_decomp (gsl_matrix * A, gsl_permutation * p, int *signum) nogil
    double gsl_linalg_LU_det (gsl_matrix * LU, int signum) nogil


cdef double determinant(gsl_matrix * matrix, size_t dimension) nogil:
    """ Computes determinant of 'matrix' but destroys it"""
    cdef int signum
    cdef gsl_permutation *p = gsl_permutation_alloc(dimension)
    gsl_linalg_LU_decomp (matrix, p, &signum)
    cdef double det = gsl_linalg_LU_det(matrix, signum)
    return det

cdef void remove_col(gsl_matrix * original, gsl_matrix * out , int nrows, int ncols, int i) nogil:

    cdef int j
    cdef gsl_matrix_view  out_view
    cdef gsl_matrix_view  original_view

    if i==0:
        original_view = gsl_matrix_submatrix(original, 0, 1, nrows, ncols-1)
        gsl_matrix_memcpy(out, &original_view.matrix)
    elif i==ncols-1:
        original_view = gsl_matrix_submatrix(original, 0, 0, nrows, ncols-1)
        gsl_matrix_memcpy(out, &original_view.matrix)
    else:
        j=i+1
        # Copy to the left of i
        original_view = gsl_matrix_submatrix(original, 0, 0, nrows, j)
        out_small_view = gsl_matrix_submatrix(out, 0, 0, nrows, j)
        gsl_matrix_memcpy(&out_small_view.matrix, &original_view.matrix)
        # Copy to the right of i
        original_view = gsl_matrix_submatrix(original, 0, j, nrows, ncols-j)
        out_small_view = gsl_matrix_submatrix(out, 0, j-1, nrows, ncols-j)
        gsl_matrix_memcpy(&out_small_view.matrix, &original_view.matrix)

@cython.boundscheck(False)
cdef void circumsphere(double[:,:] vertices, double[:] output) nogil:

    cdef gsl_matrix * D_mat = gsl_matrix_alloc(4, 5)
    cdef gsl_matrix * dummy_mat = gsl_matrix_alloc(4, 4)
    cdef double * dets = <double *> malloc(5 * sizeof(double))
    cdef double tmp
    cdef Py_ssize_t i, j,k
    cdef double sq
    cdef double elem
    
    # create matrix a
    for i in range(4):
        sq=0
        for j in range(3):
            elem = vertices[i,j]
            gsl_matrix_set(D_mat, i, j+1, elem)
            sq+=elem*elem
        gsl_matrix_set(D_mat, i, 4, 1)
        gsl_matrix_set(D_mat, i, 0, sq)
    
    
    
    for i in range(5):
        remove_col(D_mat, dummy_mat, 4, 5, i)
        dets[i] = determinant(dummy_mat, 4)
        #printf("%lf ", dets[i])
    #printf("\n")
    dets[2]*=-1
    
    gsl_matrix_free(dummy_mat)
    gsl_matrix_free(D_mat)
    sq=0
    for i in range(3):
        output[i] = dets[i+1]/(2*dets[0])
        sq+=dets[i+1]*dets[i+1]
    if dets[0] !=0 :
        output[3] = sqrt((sq-4*dets[0]*dets[4]) / (4 * dets[0] * dets[0]) )
    else:
        output[3]=0

    free(dets)



    
def single_circumsphere(double[:,:] vertices):

    out = np.empty(4, dtype=np.double)
    circumsphere(vertices, out)
    return out

def get_void_catalog(double[:,:,:] vertices, double[:,:] output, int n_simplices):

    cdef Py_ssize_t i

    for i in range(n_simplices):

        circumsphere(vertices[i, :, :], output[i, :])
    
def get_void_catalog_parallel(double[:,:,:] vertices, double[:,:] output, int n_simplices):

    cdef Py_ssize_t i

    for i in prange(n_simplices, nogil=True):

        circumsphere(vertices[i, :, :], output[i, :])

def c_ascii_writer(double [:,:] oarr, int n_elem, str oname):

    cdef FILE *fp
    cdef bytes oname_bytes = oname.encode()
    cdef char* oname_c = oname_bytes

    fp = fopen(oname_c, "w")

    cdef Py_ssize_t i

    for i in range(n_elem):
        fprintf(fp, "%lf %lf %lf %lf\n", oarr[i,0], oarr[i,1], oarr[i,2], oarr[i,3])
    
    fclose(fp)