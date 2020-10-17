import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from bin.helpers import single_circumsphere, get_void_catalog, get_void_catalog_parallel, c_ascii_writer
import os
import sys
import argparse

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
            lower[:,i] += box_size
            points = np.append(points, lower, axis=0)
            higher = points[points[:,i] >= high_range - cpy_range]
            higher[:,i] -= box_size
            points = np.append(points, higher, axis=0)
        print(points.shape)
        del lower, higher
    print(f"==> Number of tracers: {points.shape[0]}")
    print(f"==> Building Delaunay Triangulation")
    tess = Delaunay(points)
    simplex_coords=tess.points[tess.simplices[:,:], :]
    n_simplices = simplex_coords.shape[0]
    print(f"==> Found {n_simplices} simplices", flush=True)
    result = np.empty((n_simplices, 4), dtype=np.double)
    print(f"==> Computing centers and radii")
    get_void_catalog(simplex_coords, result, n_simplices)
    print(result)
    if is_box:
        for i in range(3):
            mask = (result[:,i] > low_range - out_range) & (result[:,i] < high_range + out_range)
            result = result[mask]
    print(f"==> Saving")
    #c_ascii_writer(result, n_simplices, 'tests/voids_pydive.dat')
    np.savetxt('tests/voids_pydive.dat', result, fmt="%.8f")
    #np.save('tests/voids_pydive.npy', result)
    print("==> Finished successfully")

