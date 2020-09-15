from libc.stdlib cimport malloc, free
from libc.stdio cimport FILE, fprintf, fopen, fclose, printf
from bin.CGAL.CGAL_Kernel import Point_3
from bin.CGAL.CGAL_Kernel import Sphere_3
from bin.CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3
import numpy as np



def cgal_delaunay_3(double[:,:] points, int size):

    cdef object T = Delaunay_triangulation_3()
    cdef Py_ssize_t i

    cdef object[:] point_list = np.empty(size, dtype=object)
    

    for i in range(size):
        p = Point_3(points[i,0], points[i,1], points[i,2])
        point_list[i]=p
    
    T.insert(point_list)

    return T

def delaunay_t_to_catalog(object T):
    
    