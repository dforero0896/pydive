import numpy as np
import pandas as pd
from bin.CGAL.CGAL_Kernel import Point_3
from bin.CGAL.CGAL_Kernel import Sphere_3
from bin.CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3
from bin.helpers import single_circumsphere, get_void_catalog, get_void_catalog_parallel, c_ascii_writer
from bin.cgal_helpers import cgal_delaunay_3
import os
import sys
import argparse
import math

if __name__=='__main__':

    data_fn = "tests/points.dat"
    is_box=True
    points = pd.read_csv(data_fn, delim_whitespace=True, engine='c', names=['x', 'y', 'z'], usecols=[0,1,2]).values

    if is_box:
        box_size=2500
        low_range = 0
        high_range = box_size
        cpy_range = 80
        out_range=0
        for i in range(3):
            lower = points[points[:,i] < low_range + cpy_range]
            lower[:,i] + box_size
            points = np.append(points, lower, axis=0)
            higher = points[points[:,i] >= high_range - cpy_range]
            higher[:,i] -= box_size
            points = np.append(points, higher, axis=0)
        print(points.shape)
        del lower, higher
    print(f"==> Number of tracers: {points.shape[0]}")
    print(f"==> Building Delaunay Triangulation")
    
    point_list=list(map(lambda p: Point_3(*p), points))
    
    T = Delaunay_triangulation_3()
    T.insert(point_list)
    n_simplices = T.number_of_finite_cells()
    print(f"==> Found {n_simplices} simplices", flush=True)
    result = np.empty((n_simplices, 4), dtype=np.double)
    vertices = [0,0,0,0]
    print(f"==> Computing centers and radii")
    for i, cell in enumerate(T.finite_cells()):

        vertices[0] = cell.vertex(0).point()
        vertices[1] = cell.vertex(1).point()
        vertices[2] = cell.vertex(2).point()
        vertices[3] = cell.vertex(3).point()
        sphere = Sphere_3(*vertices)
        result[i,0]=sphere.center().x()
        result[i,1]=sphere.center().y()
        result[i,2]=sphere.center().z()
        result[1,3] = math.sqrt(sphere.squared_radius())

    
    
    print(result)
    if is_box:
        for i in range(3):
            mask = (result[:,i] > low_range - out_range) & (result[:,i] < high_range + out_range)
            result = result[mask]
    print(f"==> Saving")
    #c_ascii_writer(result, n_simplices, 'tests/voids_pydive.dat')
    np.savetxt('tests/voids_pydive_cgal.dat', result, fmt="%.8f")
    #np.save('tests/voids_pydive.npy', result)
    print("==> Finished successfully")

