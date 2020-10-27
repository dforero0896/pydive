import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from bin.helpers import single_circumsphere, get_void_catalog, get_void_catalog_parallel, c_ascii_writer
import os
import sys
import argparse

def pydive(points, r_min, r_max, box_size, is_box=False, cpy_range=80):

    print(f"==> {points.shape[0]} tracers found.")
    if is_box:
        low_range = 0
        high_range = box_size
        out_range=0
        print("==> Duplicating boundaries for periodic condition", flush=True)
        for i in range(3):
            lower = points[points[:,i] < low_range + cpy_range]
            lower[:,i] += box_size
            points = np.append(points, lower, axis=0) #This is not memory efficient.
            higher = points[points[:,i] >= high_range - cpy_range]
            higher[:,i] -= box_size
            points = np.append(points, higher, axis=0)
        del lower, higher
    print(f"==> Number of vertices: {points.shape[0]}")
    print(f"==> Building Delaunay Triangulation")   
    tess = Delaunay(points)
    del points
    simplex_coords=tess.points[tess.simplices[:,:], :]
    del tess
    n_simplices = simplex_coords.shape[0]
    print(f"==> Found {n_simplices} simplices", flush=True)
    result = np.empty((n_simplices, 4), dtype=np.double)
    print(f"==> Computing centers and radii")
    get_void_catalog(simplex_coords, result, n_simplices)
    del simplex_coords
    #get_void_catalog_parallel(simplex_coords, result, n_simplices)
    result=result.astype(np.float32)
    if is_box:
        for i in range(3):
            mask = (result[:,i] > low_range - out_range) & (result[:,i] < high_range + out_range)
            result = result[mask]
    # Radius mask
    mask = (result[:,3] > r_min) & (result[:,3] < r_max)
    result = result[mask] 
    print("==> Finished DIVE call successfully")
    return result

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--input_catalog", default = "tests/points.dat")
    parser.add_argument("-out", "--output_catalog", default = "tests/voids_pydive.dat")
    parser.add_argument("-box", "--box_size", dest="box_size", default=2500, type=np.float64)
    parser.add_argument("-rmin", "--rmin", dest="rmin", default=0, type=np.float64)
    parser.add_argument("-rmax", "--rmax", dest="rmax", default=999, type=np.float64)
    args = parser.parse_args()
    data_fn = args.input_catalog
    is_box=True
    box_size=2500
    print(f"==> Reading file {data_fn}", flush=True)
    points = pd.read_csv(data_fn, delim_whitespace=True, engine='c', names=['x', 'y', 'z'], usecols=[0,1,2]).values.astype(np.float64)
    result = pydive(points, args.rmin, args.rmax, box_size, is_box, cpy_range=80)
    print(f"==> Saving voids with radius ({args.rmin}, {args.rmax})")
    #c_ascii_writer(result, n_simplices, 'tests/voids_pydive.dat')
    np.savetxt(args.output_catalog, result, fmt="%.8f")
    #np.save('tests/voids_pydive.npy', result)
    print("==> Finished successfully")