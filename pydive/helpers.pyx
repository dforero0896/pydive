#cython: language_level=3 
import cython
from cython.parallel import prange, threadid
cimport openmp
import numpy as np
from libc.stdlib cimport malloc, free
from libc.stdio cimport FILE, fprintf, fopen, fclose, printf

#################################################### Definitions ###########################################################

DEF SPEED_OF_LIGHT=299792.458
DEF PI=3.1415926535
DEF PREC_DIGIT=10

cdef extern from "math.h":
    double sqrt(double x) nogil
    double cos(double x) nogil
    double sin(double x) nogil
    double pow(double x, double y) nogil
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
                             const size_t n2) nogil
    gsl_matrix *  gsl_matrix_alloc(size_t n1, size_t n2) nogil
    double  gsl_matrix_get(gsl_matrix * m, size_t i, size_t j) nogil
    void  gsl_matrix_set(gsl_matrix * m, size_t i, size_t j, double x) nogil
    void  gsl_matrix_free(gsl_matrix * m) nogil
    gsl_matrix_view  gsl_matrix_submatrix(gsl_matrix * m, size_t k1, size_t k2, size_t n1, size_t n2) nogil
    int  gsl_matrix_memcpy(gsl_matrix * dest, gsl_matrix * src) nogil

cdef extern from "gsl/gsl_math.h":
    ctypedef struct gsl_function:
        double (* function) (double x, void * params) 
        void * params
cdef extern from "gsl/gsl_integration.h":
    ctypedef struct gsl_integration_workspace:
        size_t limit
        size_t nrmax
        size_t i
        size_t maximum_level
        double *alist
        double *blist
        double *rlist
        double *elist
        size_t *order
        size_t *level
    gsl_integration_workspace * gsl_integration_workspace_alloc(const size_t n) nogil
    void gsl_integration_workspace_free(gsl_integration_workspace * w) nogil
    int gsl_integration_qag (const gsl_function *f, 
                         double a, double b,
                         double epsabs, double epsrel, size_t limit,
                         int key,
                         gsl_integration_workspace * workspace,
                         double *result, double *abserr) nogil
    DEF GSL_INTEG_GAUSS51 = 5
    DEF GSL_INTEG_GAUSS61 = 6



cdef extern from "gsl/gsl_linalg.h":
    int gsl_linalg_LU_decomp (gsl_matrix * A, gsl_permutation * p, int *signum) nogil
    double gsl_linalg_LU_det (gsl_matrix * LU, int signum) nogil


######################################### DT void catalog computations ##############################################

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


def get_void_catalog(double[:,:,:] vertices, double[:,:] output, int n_simplices):

    cdef Py_ssize_t i

    for i in range(n_simplices):

        circumsphere(vertices[i, :, :], output[i, :])
    
def get_void_catalog_parallel(double[:,:,:] vertices, double[:,:] output, int n_simplices, int n_threads):

    cdef Py_ssize_t i
    for i in prange(n_simplices, nogil=True, num_threads=n_threads):
        circumsphere(vertices[i, :, :], output[i, :])

########################################## Sky to Cartesian Coordinate conversion ################################################

@cython.boundscheck(False)
cdef double comoving_dist_integrand(double x, void * params) nogil:

    # Signature must match what GSL asks for in the integration.
    # Extract parameters
    cdef double H0 =  (<double *> params)[0]
    cdef double OmegaL =  (<double *> params)[1]
    cdef double OmegaM =  (<double *> params)[2]
    cdef double c =  (<double *> params)[3]

    cdef double H = H0 * sqrt(OmegaL + OmegaM * pow( 1 + x , 3))

    return c / H
    
cdef double integrate_z(double z, double H0, double OmegaL, double OmegaM) nogil :

    cdef gsl_integration_workspace *w
    cdef double result, error
    cdef double c = SPEED_OF_LIGHT
    cdef int prec = PREC_DIGIT
    cdef gsl_function integrand
    cdef double* params = [ H0, OmegaL, OmegaM, c ]

    integrand.function = &comoving_dist_integrand
    integrand.params = params

    w = gsl_integration_workspace_alloc(1000)
    gsl_integration_qag(&integrand, 0, z, 0, pow(10, -prec), 1000, GSL_INTEG_GAUSS51, w, &result, &error)
    gsl_integration_workspace_free(w)

    return result

@cython.boundscheck(False)
def sky_to_cart_parallel(double[:,:] input, double[:,:] output, int n_lines, int n_threads, double H0=67.7, double OmegaM=0.307115):

    cdef double OmegaL = 1 - OmegaM
    cdef double dist, ra, dec, h
    h = H0 / 100
    cdef Py_ssize_t i
    for i in prange(n_lines, nogil=True, num_threads=n_threads):
        dist = integrate_z(input[i,2], H0, OmegaL, OmegaM)
        ra = input[i,0] * PI / 180
        dec = input[i,1] * PI / 180
        #X
        output[i,0] = dist * cos(dec) * cos(ra) * h
        #Y
        output[i,1] = dist * cos(dec) * sin(ra) * h
        #Z
        output[i,2] = dist * sin(dec) * h
        

def c_ascii_writer(double [:,:] oarr, int n_elem, str oname):

    cdef FILE *fp
    cdef bytes oname_bytes = oname.encode()
    cdef char* oname_c = oname_bytes

    fp = fopen(oname_c, "w")

    cdef Py_ssize_t i

    for i in range(n_elem):
        fprintf(fp, "%lf %lf %lf %lf\n", oarr[i,0], oarr[i,1], oarr[i,2], oarr[i,3])
    
    fclose(fp)
