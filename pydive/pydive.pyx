#cython: language_level=3 
#cython: profile=True 
#cython: boundscheck=True
import cython
from cython.parallel import prange, threadid
cimport openmp
import numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport fabs
from libc.stdio cimport FILE, fprintf, fopen, fclose, printf, fflush, stdout

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


######################################### DT void catalog computations ##############################################

cdef double determinant(gsl_matrix * matrix, size_t dimension) nogil:
    """ Computes determinant of 'matrix' but destroys it"""
    cdef int signum
    cdef gsl_permutation *p = gsl_permutation_alloc(dimension)
    gsl_linalg_LU_decomp (matrix, p, &signum)
    cdef double det = gsl_linalg_LU_det(matrix, signum)
    gsl_permutation_free(p)
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


cdef double simplex_volume(double[:,:] vertices) nogil:

    # vertices has shape (nvertices, ndims) = (4, 3)
    cdef gsl_matrix * matrix = gsl_matrix_alloc(4, 4)
    cdef double volume, elem
    cdef Py_ssize_t i, j

    for i in range(3):
        for j in range(4):
            elem = vertices[j,i]
            gsl_matrix_set(matrix, i, j, elem)
            gsl_matrix_set(matrix, 3, j, 1)
    
    volume = fabs(determinant(matrix, 4) / 6)

    gsl_matrix_free(matrix)

    return volume

cdef double facet_area(double[:] vectora, double[:] vectorb) nogil:
    cdef double x, y, z
    x = vectora[1]*vectorb[2] - vectora[2]*vectorb[1]
    y = - (vectora[0]*vectorb[2] - vectora[2]*vectorb[0])
    z = vectora[0]*vectorb[1] - vectora[1]*vectorb[0]

    facet_area = 0.5 * sqrt(x*x + y*y + z*z)

    return facet_area

cdef inline double sphericity(double simp_volume, double simp_area) nogil:
    
    cdef double sph = pow(PI, 1./3) * pow(6 * simp_volume, 2./3) / simp_area

    return sph


cdef double simplex_area(double[:,:] vertices) nogil:

    cdef double tot_area = 0
    cdef Py_ssize_t i, j
    tot_area += facet_area(vertices[0,:], vertices[1,:])
    tot_area += facet_area(vertices[0,:], vertices[2,:])
    tot_area += facet_area(vertices[0,:], vertices[3,:])
    tot_area += facet_area(vertices[1,:], vertices[2,:])

    return tot_area
            



cdef int circumsphere(double[:,:] vertices, double[:] output) nogil except -1:

    cdef gsl_matrix * D_mat = gsl_matrix_alloc(4, 5)
    cdef gsl_matrix * dummy_mat = gsl_matrix_alloc(4, 4)
    cdef double * dets = <double *> malloc(5 * sizeof(double))
    #cdef double dets[5]
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
        dets[i] = 0
        remove_col(D_mat, dummy_mat, 4, 5, i)
        dets[i] = determinant(dummy_mat, 4)
    dets[2]*=-1
    
    gsl_matrix_free(dummy_mat)
    gsl_matrix_free(D_mat)
    sq=0
    for i in range(3):
        output[i] = dets[i+1]/(2*dets[0])
        sq+=dets[i+1]*dets[i+1]
    
    if fabs(dets[0]) >  1e-8 :
        output[3] = sqrt((sq-4*dets[0]*dets[4]) / (4 * dets[0] * dets[0]) )
    else:
        
        output[3]=0

    free(dets)


def get_void_catalog(double[:,:,:] vertices, double[:,:] output, int n_simplices):

    cdef Py_ssize_t i    
    for i in range(n_simplices):
        circumsphere(vertices[i, :, :], output[i, :])
    
def get_void_catalog_parallel(double[:,:,:] vertices, double[:,:] output, int n_simplices, int n_threads):
    """
    Compute the void catalog (circumcenters and radii) only
    Parameters:
        vertices: ndarray of double, shape (n_simplices, 4, 3)
            Array containing the coordinates (3) of the 4 vertices
            that define the simplex.
        output: ndarray of double, shape (n_simplices, 4)
            Array to put the resulting void catalog in. Should
            contain enough space for (x, y, z, r) values.
        n_simplices: int
            Number of simplices found by the triangulation. 
            (n_simplices = vertices.shape[0])
        n_threads: int
            Number of threads to use if compiled with openmp support.

    Returns:
        None
    """
    cdef Py_ssize_t i
    for i in prange(n_simplices, nogil=True, num_threads=n_threads):
        circumsphere(vertices[i, :, :], output[i, :])
   
def get_void_catalog_dtfe(double[:,:,:] vertices, 
                            int[:,:] simplex_indices,
                            double[:] weights,
                            double[:] selection,
                            double[:,:] output,
                            double[:] volumes,
                            double[:] density,
                            int n_simplices, 
                            int n_vertices, 
                            double average_density,
                            int n_threads):
    """
    Compute the void catalog and estimate tracer density with DTFE
    (Schaap, W. E. & van de Weygaert, R. 2000)
    Parameters:
        vertices: ndarray of double, shape (n_simplices, 4, 3)
            Array containing the coordinates (3) of the 4 vertices
            that define the simplex.
        simplex_indices: ndarray of int, shape (n_simplices, 4)
            Array containing the indices of the vertices that
            define the simplex.
        weights: ndarray of double, shape (n_vertices,)
            Array containing the weights of each tracer.
        selection: ndarray of double, shape (n_vertices,)
            Array containing the values of the selection function 
            at each vertex position.
        output: ndarray of double, shape (n_simplices, 5)
            Array to put the resulting void catalog in. Should
            contain enough space for (x, y, z, r, v) values.
        volumes: ndarray of double, shape (n_vertices,)
            Array to output the volume of the contiguous Voronoi
            cell for each vertex. 
        density: ndarray of double, shape (n_vertices,)
            Array to output the estimated local density for each
            vertex.
        n_simplices: int
            Number of simplices found by the triangulation. 
            (n_simplices = vertices.shape[0])
        n_vertices: int
            Number of vertices passed to the triangulation. 
        n_threads: int
            Number of threads to use if compiled with openmp support.

    Returns:
        None
    """

    cdef Py_ssize_t i, k, vertex_index, l, m
    cdef double volume=0
    cdef double factor = 4. / average_density
    #for i in prange(n_simplices, nogil=True, num_threads=n_threads):
    for i in range(n_simplices):
        circumsphere(vertices[i, :, :], output[i, :])
        volume = simplex_volume(vertices[i, :, :])
        output[i,4] = volume
        for k in range(4):
            vertex_index = simplex_indices[i,k]
            volumes[vertex_index] += volume #race condition 
    for i in prange(n_vertices, nogil=True, num_threads=n_threads):
        if volumes[i] > 0:
            density[i] = 4. * weights[i] / (average_density * selection[i] * volumes[i])
            #density[i] = factor / (volumes[i])
        #else:
        #    density[i] = -99999
        

def get_satellites(double[:,:] data, 
                    int[:,:,:] grid_void_count, 
                    int[:,:,:,:] grid_id_buffer, 
                    int[:] central_id, 
                    bint[:] is_central, 
                    int[:] n_satellites, 
                    double[:] distances, 
                    double[:] central_radius, 
                    double box_size):

    """
    Compute the number of satellites from a void catalog

    Parameters:

    data: ndarray shape (nvoids, 4)
        Array containing the void coordinates and radii (x,y,z,r)
    grid_void_count: ndarray shape (ngrid, ngrid, ngrid)
        Array containing the number of voids in the cell (ix, iy, iz)
        See func. allocate_to_grid
    grid_id_buffer: ndarray (ngrid, ngrid, ngrid, nbuffer)
        Array containing the ids of the voids in the cell (ix, iy, iz)
        See func. allocate_to_grid
    central_id: ndarray (nvoids,)
        Array to store the id of the central void
    is_central: ndarray (nvoids,)
        Array to store boolean value (True if central)
    n_satellites: ndarray (nvoids,)
        Array to store number of satellites for a central void,
        zero if satellite.
    distances: ndarray (nvoids,)
        Array to store de distance to the corresponding central void.
    central_radius: ndarray (nvoids,)
        Array to store the radius of the corresponding central void.
    box_size: double
        Size of the box in which the grid is embedded
    
    """

    cdef int n_grid = grid_void_count.shape[0]
    cdef double inv_bin_size = n_grid / box_size
    cdef int xmin, xmax, ymin, ymax, zmin, zmax
    cdef Py_ssize_t i, j, k
    cdef double sqr, r

    for i in range(data.shape[0]):
        
        if is_central[i]:
            sqr = data[i,3]*data[i,3]
            central_id[i] = i
            central_radius[i] = data[i,3]
            r = data[i,3] 
            xmin = <int> (((data[i,0] - r) * inv_bin_size) )
            xmax = <int> (((data[i,0] + r) * inv_bin_size) )
            ymin = <int> (((data[i,1] - r) * inv_bin_size) )
            ymax = <int> (((data[i,1] + r) * inv_bin_size) )
            zmin = <int> (((data[i,2] - r) * inv_bin_size) )
            zmax = <int> (((data[i,2] + r) * inv_bin_size) )
            
            for xid in range(xmin, xmax+1):
                if xid >= n_grid: continue
                for yid in range(ymin, ymax+1):
                    if yid >= n_grid: continue
                    for zid in range(zmin, zmax+1):
                        if zid >= n_grid: continue
                        
                        for j in range(grid_void_count[xid, yid, zid]):
                            k = grid_id_buffer[xid, yid, zid, j]
                            
                            
                            if k <= i: continue
                            
                            distance = (data[i, 0] - data[k, 0])**2 + (data[i, 1] - data[k, 1])**2 + (data[i, 2] - data[k, 2])**2
                            
                            if distance < sqr:
                                
                                if is_central[k]:    
                                    central_id[k] = i
                                    central_radius[k] = data[i,3]
                                    is_central[k] = False
                                    n_satellites[i] += 1
                                    distances[k] = distance
                                elif not is_central[k] and distances[k] < distance:
                                    n_satellites[central_id[k]] -= 1
                                    central_id[k] = i
                                    central_radius[k] = data[i,3]
                                    is_central[k] = False
                                    n_satellites[i] += 1
                                    distances[k] = distance






def save_void_catalog(double[:,:,:] vertices, double[:] output, int n_simplices, str oname, double r_min, 

                        double r_max, bint is_box, float low_range, float box_size, bint volume):

    """ 
    Save to ascii file directly as circumspheres are computed.
    Useful when saving to named pipes for reading with other codes.
    If there is need to use void data in python, do not use this function. 
    
    Compute the void catalog (circumcenters and radii) only
    Parameters:
        vertices: ndarray of double, shape (n_simplices, 4, 3)
            Array containing the coordinates (3) of the 4 vertices
            that define the simplex.
        output: ndarray of double, shape (4,) ((5,) if volume==True)
            Array to put the resulting void information in. Should
            contain enough space for (x, y, z, r) ((x, y, z, r, v)) 
            values.
        n_simplices: int
            Number of simplices found by the triangulation. 
            (n_simplices = vertices.shape[0])
        str: oname
            Output path where to write void catalog
        r_min: double
            Minimum void radius to write
        r_max: double
            Maximum void radius to write
        is_box: bool
            Boolean flag defining if catalog is to be treated as
            as a box.
        low_range: float
            Minimum x_i = (x, y, z) value.
        box_size: float
            Box size.
        volume: bool
            Boolean flag defining if simplex volumes should be written.
      
    Returns:
        None
    """
    
    if volume and <int> output.shape[0] < 5:
        raise ValueError("Output buffer not long enough to output simplex volume")


    cdef float high_range  = low_range + box_size
    cdef float out_range = 0

    cdef FILE *fp
    cdef bytes oname_bytes = oname.encode()
    cdef char* oname_c = oname_bytes

    fp = fopen(oname_c, "w")

    cdef Py_ssize_t i, j
    cdef int counter=0
    printf("==> Saving voids with radius in (%lf, %lf)\n", r_min, r_max)
    fflush(stdout)
    for i in range(n_simplices):
        circumsphere(vertices[i, :, :], output[:])
        if volume:
            output[4] = simplex_volume(vertices[i, :, :])
        if (output[3] > r_min) and (output[3] < r_max):
            if is_box:
                if (output[0] > low_range-out_range) and (output[0] < high_range+out_range) and (output[1] > low_range-out_range) and (output[1] < high_range+out_range) and (output[2] > low_range-out_range) and (output[2] < high_range+out_range):
                    counter+=1
                    if not volume:
                        fprintf(fp, "%lf %lf %lf %lf\n", output[0], output[1], output[2], output[3])                  
                    else:
                        fprintf(fp, "%lf %lf %lf %lf %lf\n", output[0], output[1], output[2], output[3], output[4])                  
            else:
                counter+=1
                if not volume:
                    fprintf(fp, "%lf %lf %lf %lf\n", output[0], output[1], output[2], output[3])                  
                else:
                    fprintf(fp, "%lf %lf %lf %lf %lf\n", output[0], output[1], output[2], output[3], output[4])                  

    
    
    fclose(fp)
    printf("==> Done, saved %i voids.\n", counter)
    fflush(stdout)





def interpolate_at_circumcenters(double[:] density_at_vertices,
                                 double[:,:] vertex_coordinates,
                                 int[:,:] simplex_indices,
                                 double[:,:] circumcenters,
                                 double[:] density_at_voids,
                                 int interp_kind,
                                 int n_threads):

    """
    Interpolates the density field at the void positions.

    Parameters:
        density_at_vertices: ndarray of double, shape (n_vertices,)
            Array containing the field values at the vertex positions
        vertex_coordinates: ndarray of double, shape (n_vertices, 3)
            Array containing the coordinates of points used for the 
            tesselation.
        simplex_indices: ndarray of int, shape (n_simplices, 4)
            Array containing the indices in vertex_coordinates of the 
            points that define each simplex.
        circumcenters: ndarray of double, shape (n_simplices, 4)
            Array containing the coordinates of each circumcenter and
            the void radius.
        density_at_voids: ndarray of double, shape (n_simplices,)
            Array to output the interpolated densities.
        interp_kind: int 0 or 1
            If 0 IDW is used for each simplex. If 1, linear interpolation
            is performed.
        n_threads: int
            Number of threads to use if compiled with openmp support.
        
    Returns:
        None
    """

    cdef Py_ssize_t i
    cdef double value_at_void
    assert(interp_kind<2)
    #for i in prange(simplex_indices.shape[0], nogil=True, num_threads=n_threads):
    for i in range(simplex_indices.shape[0]):
        if interp_kind==1:
            value_at_void = interpolate_at_circumcenter_linear(density_at_vertices,
                                                        vertex_coordinates,
                                                        simplex_indices[i,:],
                                                        circumcenters[i,:])
        elif interp_kind == 0:
            value_at_void = interpolate_at_circumcenter_idw(density_at_vertices,
                                                    vertex_coordinates,
                                                    simplex_indices[i,:],
                                                    circumcenters[i,:],
                                                    2)


        density_at_voids[i] = value_at_void

cdef double interpolate_at_circumcenter_linear(double[:] density_at_vertices,
                                 double[:,:] vertex_coordinates,
                                 int[:] simplex_indices,
                                 double[:] circumcenter) nogil except -1:
        
    cdef Py_ssize_t i, j, k
    #for i in prange(n_vertices, nogil=True, num_threads=n_threads):
    cdef gsl_matrix * A = gsl_matrix_alloc(3, 3)
    cdef gsl_vector * nablaf = gsl_vector_alloc(3)
    cdef gsl_vector * f = gsl_vector_alloc(3)
    cdef gsl_vector * dx = gsl_vector_alloc(3)
    cdef int s
    cdef double elem
    cdef double value_at_void
    cdef gsl_permutation * p = gsl_permutation_alloc (3);
    
    cdef int vertex_id = 1

    for j in range(3): # Iterate over vertices, rows
        gsl_vector_set(dx, j, circumcenter[j] - vertex_coordinates[simplex_indices[vertex_id],j])
        gsl_vector_set(f, j, density_at_vertices[simplex_indices[j+1]] - density_at_vertices[simplex_indices[vertex_id]])
        for k in range(3): # Iterate over dimensions, cols
            elem = vertex_coordinates[simplex_indices[j+1], k] - vertex_coordinates[simplex_indices[0], k]
            gsl_matrix_set(A, j, k, elem)
    

    gsl_linalg_LU_decomp (A, p, &s);
    gsl_linalg_LU_solve (A, p, f, nablaf);


    gsl_blas_ddot(nablaf, dx, &value_at_void)
    value_at_void+=density_at_vertices[simplex_indices[vertex_id]]



    gsl_permutation_free (p);
    gsl_matrix_free(A)
    gsl_vector_free(nablaf)
    gsl_vector_free(f)
    gsl_vector_free(dx)

    return value_at_void
cdef double interpolate_at_circumcenter_idw(double[:] density_at_vertices,
                                            double[:,:] vertex_coordinates,
                                            int[:] simplex_indices,
                                            double[:] circumcenter,
                                            float p) nogil except -1:
                    
    cdef Py_ssize_t i, j, k
    cdef double elem
    cdef double value_at_void
    cdef double numerator = 0    
    cdef double weight
    if circumcenter[3] > 1e-8:
        weight = 1. / circumcenter[3]**p
    else:
        weight = 0
    for j in range(4): # Iterate over vertices, rows
        numerator += density_at_vertices[simplex_indices[j]] * weight
    
    if weight > 1e-8:
        value_at_void = numerator / (4 * weight)
    else:
        value_at_void =0
        

    return value_at_void
########################################## Sky to Cartesian Coordinate conversion ################################################

@cython.boundscheck(False)
cdef double comoving_dist_integrand(double x, void * params) nogil:

    # Signature must match what GSL asks for in the integration.
    # Extract parameters
    cdef double H0 =  (<double *> params)[0]
    cdef double OmegaL =  (<double *> params)[1]
    cdef double OmegaM =  (<double *> params)[2]
    cdef double c =  (<double *> params)[3]
    #printf("%lf %lf %lf %lf\n", H0, OmegaL, OmegaM, c)

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
        

def c_ascii_writer_double(double [:,:] oarr, int n_elem, str oname):

    cdef FILE *fp
    cdef bytes oname_bytes = oname.encode()
    cdef char* oname_c = oname_bytes

    fp = fopen(oname_c, "w")

    cdef Py_ssize_t i

    for i in range(n_elem):
        fprintf(fp, "%lf %lf %lf %lf\n", oarr[i,0], oarr[i,1], oarr[i,2], oarr[i,3])
    
    fclose(fp)

def c_ascii_writer_single(float [:,:] oarr, int n_elem, str oname):

    cdef FILE *fp
    cdef bytes oname_bytes = oname.encode()
    cdef char* oname_c = oname_bytes

    fp = fopen(oname_c, "w")

    cdef Py_ssize_t i

    for i in range(n_elem):
        fprintf(fp, "%f %f %f %f\n", oarr[i,0], oarr[i,1], oarr[i,2], oarr[i,3])
    
    fclose(fp)


cpdef int allocate_to_grid(double[:,:] data, 
                    int[:,:,:] grid_void_count, 
                    int[:,:,:,:] grid_id_buffer, 
                    double box_size,
                    int n_threads) nogil except -1:
    
    cdef int n_grid = grid_void_count.shape[0]
    cdef double bin_size = box_size / n_grid
    cdef double inv_bin_size = n_grid / box_size
    cdef int n_buffer = grid_id_buffer.shape[3]
    cdef int idx, idy, idz
    cdef int i
    for i in prange(data.shape[0], nogil=True, num_threads=n_threads):
        
        idx = <int> ((data[i, 0] * inv_bin_size) + n_grid) % n_grid
        idy = <int> ((data[i, 1] * inv_bin_size) + n_grid) % n_grid
        idz = <int> ((data[i, 2] * inv_bin_size) + n_grid) % n_grid
        
        if n_buffer > grid_void_count[idx, idy, idz]: return -1
        grid_id_buffer[idx, idy, idz, grid_void_count[idx, idy, idz]] = i
        grid_void_count[idx, idy, idz] += 1
    
    return 0