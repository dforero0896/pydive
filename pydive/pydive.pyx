#cython: language_level=3 
#cython: profile=True 
#cython: boundscheck=False
#cython: wraparound=False
import cython
from cython.parallel import prange, threadid
cimport openmp
import numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport fabs
from libc.stdio cimport FILE, fprintf, fopen, fclose, printf, fflush, stdout
from libcpp.vector cimport vector

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
    void gsl_permutation_free(gsl_permutation * p) nogil
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

cdef extern from "gsl/gsl_vector.h":
    ctypedef struct gsl_vector:
        size_t size
        size_t stride
        double * data
        gsl_block * block
        int owner
    gsl_vector *gsl_vector_alloc (const size_t n) nogil
    void gsl_vector_set (gsl_vector * v, const size_t i, double x) nogil
    void gsl_vector_free (gsl_vector * v) nogil
cdef extern from "gsl/gsl_blas.h":
    int gsl_blas_ddot (const gsl_vector * X,
                   const gsl_vector * Y,
                   double * result
                   ) nogil
cdef extern from "gsl/gsl_math.h":
    ctypedef struct gsl_function:
        double (* function) (double x, void * params) 
        void * params
cdef extern from "gsl/gsl_linalg.h":
    int gsl_linalg_LU_decomp (gsl_matrix * A, gsl_permutation * p, int *signum) nogil
    int gsl_linalg_LU_solve (const gsl_matrix * LU,
                         const gsl_permutation * p,
                         const gsl_vector * b,
                         gsl_vector * x) nogil
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


cdef extern from "delaunay_backend.cpp":
    cdef cppclass DelaunayOutput:
        vector[double] x, y, z, r, volume, dtfe, area
        vector[size_t] vertices[4]
        size_t n_simplices
    DelaunayOutput cdelaunay(vector[double] X, vector[double] Y, vector[double] Z) nogil
    DelaunayOutput cdelaunay_periodic_extend(vector[double] X, vector[double] Y, vector[double] Z) nogil
    DelaunayOutput cdelaunay_periodic(vector[double] X, vector[double] Y, vector[double] Z) nogil
    DelaunayOutput cdelaunay_full(vector[double] X, vector[double] Y, vector[double] Z) nogil
    
def get_void_catalog_cgal(double[:,:] points,
                        bint periodic=False,
                        ):

    cdef Py_ssize_t i,k
    cdef vector[double] in_x, in_y, in_z
    
    
    
    in_x.reserve(points.shape[0])
    in_y.reserve(points.shape[0])
    in_z.reserve(points.shape[0])
    
    for i in range(points.shape[0]):
        in_x.push_back(points[i,0])
        in_y.push_back(points[i,1])
        in_z.push_back(points[i,2])
    
    cdef DelaunayOutput voids
    if not periodic:
        voids = cdelaunay_periodic(in_x, in_y, in_z)
    else:
        voids = cdelaunay_periodic_extend(in_x, in_y, in_z)
        
    cdef size_t n_simplices
    in_x.clear()
    in_y.clear()
    in_z.clear()
    
    n_simplices = voids.n_simplices
    output = np.zeros((n_simplices, 4), dtype=np.double)
    for k in range(n_simplices):
        output[k,0] = voids.x[k]
        output[k,1] = voids.y[k]
        output[k,2] = voids.z[k]
        output[k,3] = voids.r[k]
    
    

    return output

@cython.wraparound(False)
@cython.boundscheck(False)
def get_void_catalog_full(double[:,:] points,
                        bint periodic=False,
                        ):

    cdef Py_ssize_t i,k
    cdef vector[double] in_x, in_y, in_z
    
    
    
    in_x.reserve(points.shape[0])
    in_y.reserve(points.shape[0])
    in_z.reserve(points.shape[0])
    
    for i in range(points.shape[0]):
        in_x.push_back(points[i,0])
        in_y.push_back(points[i,1])
        in_z.push_back(points[i,2])
    
    cdef DelaunayOutput voids
    if not periodic:
        voids = cdelaunay_full(in_x, in_y, in_z)
    else:
        voids = cdelaunay_periodic_extend(in_x, in_y, in_z)
        
    cdef size_t n_simplices
    in_x.clear()
    in_y.clear()
    in_z.clear()
    
    cdef double p, w, numerator
    


    n_simplices = voids.n_simplices
    output = np.zeros((n_simplices, 7), dtype=np.double)
    dtfe = np.zeros((points.shape[0], 1), dtype=np.double)

    printf("==> Computing DTFE\n")
    fflush(stdout)
    for k in range(points.shape[0]):
        dtfe[k] = 4. / voids.dtfe[k]
    printf("==> Copying voids and interpolating\n")
    fflush(stdout)
    for k in range(n_simplices):
        output[k,0] = voids.x[k]
        output[k,1] = voids.y[k]
        output[k,2] = voids.z[k]
        output[k,3] = voids.r[k]
        output[k,4] = voids.volume[k]
        w = 1. / voids.r[k]
        numerator = 0
        for i in range(4):
            numerator += w * dtfe[<size_t> voids.vertices[i][k]]
        output[k,5] = numerator / (4 * w)
        output[k,6] = voids.area[k]
        


    
        
    

    return output, dtfe




