import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from helpers import get_void_catalog, get_void_catalog_parallel, sky_to_cart_parallel, save_void_catalog, get_void_catalog_volumes, get_void_catalog_volumes_parallel
import os
import sys
import gc

def extend_boundaries_box(points, box_size=2500, cpy_range=80, low_range=0):
    high_range=low_range+box_size
    print("==> Duplicating boundaries for periodic condition", flush=True)
    for i in range(3):
        lower = points[points[:,i] < low_range + cpy_range]
        lower[:,i] += box_size
        points = np.append(points, lower, axis=0) #This is not memory efficient.
        higher = points[points[:,i] >= high_range - cpy_range]
        higher[:,i] -= box_size
        points = np.append(points, higher, axis=0)
    del lower, higher
    return points


def galaxies_to_voids(points, r_min=0, r_max=99999, coordinate_conversion=False, box_size=2500, is_box=False, low_range=0, cpy_range=80, n_threads=64, zmin=0.2, zmax=0.75, volume=False):
    print(f"==> {points.shape[0]} tracers found.")
    if is_box:
        high_range= low_range + box_size
        out_range=0
        points = extend_boundaries_box(points, box_size=box_size, cpy_range=cpy_range)
    if coordinate_conversion:
        print(f"==> Performing sky to cartesian coordinate conversion.", flush=True)
        mask = (points[:,2] > zmin) & (points[:,2] < zmax)
        points = points[mask]
        del mask
        sky_to_cart_parallel(points, points, points.shape[0], n_threads)
    print(f"==> Number of vertices: {points.shape[0]}")
    print(f"==> Building Delaunay Triangulation", flush=True)   
    tess = Delaunay(points.astype(np.double))

    simplex_coords=tess.points[tess.simplices[:,:], :]
    del tess
    n_simplices = simplex_coords.shape[0]
    print(f"==> Found {n_simplices} simplices", flush=True)
    result = np.empty((n_simplices, 4), dtype=np.double)
    print(f"==> Computing centers and radii", flush=True)
    if n_threads == 1 or n_threads is None: 
        if not volume:
            get_void_catalog(simplex_coords.astype(np.double), result, n_simplices)
        else:
            get_void_catalog_volumes(simplex_coords.astype(np.double), result, n_simplices)
    else: 
        if not volume:
            get_void_catalog_parallel(simplex_coords.astype(np.double), result, n_simplices, n_threads)
        else:
            get_void_catalog_volumes_parallel(simplex_coords.astype(np.double), result, n_simplices, n_threads)
    del simplex_coords
    gc.collect()
    result=result.astype(np.float32)
    gc.collect()
    if is_box:
        for i in range(3):
            mask = (result[:,i] > low_range - out_range) & (result[:,i] < high_range + out_range)
            result = result[mask]
    


    # Radius mask
    mask = (result[:,3] > r_min) & (result[:,3] < r_max)
    result = result[mask] 
    print("==> Finished DIVE call successfully")
    return result


def save_galaxies_to_voids(points, oname, r_min=0, r_max=99999, coordinate_conversion=False, box_size=2500, is_box=False, low_range=0, cpy_range=80, zmin=0.2, zmax=0.75, n_threads=64, volume=False):
    print(f"==> {points.shape[0]} tracers found.")
    if is_box:
        high_range= low_range + box_size
        out_range=0
        points = extend_boundaries_box(points, box_size=box_size, cpy_range=cpy_range)
    if coordinate_conversion:
        print(f"==> Performing sky to cartesian coordinate conversion.", flush=True)
        mask = (points[:,2] > zmin) & (points[:,2] < zmax)
        points = points[mask]
        del mask
        sky_to_cart_parallel(points, points, points.shape[0], n_threads)
    print(f"==> Number of vertices: {points.shape[0]}")
    print(f"==> Building Delaunay Triangulation", flush=True)   
    tess = Delaunay(points.astype(np.double))

    simplex_coords=tess.points[tess.simplices[:,:], :]
    del tess
    n_simplices = simplex_coords.shape[0]
    print(f"==> Found {n_simplices} simplices", flush=True)
    print(f"==> Computing centers and radii", flush=True)
    if volume:
        buffer = np.zeros((5,), dtype=np.double)
    else:
        buffer = np.zeros((4,), dtype=np.double)
    save_void_catalog(simplex_coords.astype(np.double), buffer, n_simplices, oname, r_min, r_max, is_box, low_range, box_size, volume)
    del simplex_coords
    gc.collect()
      
    print("==> Finished DIVE call successfully")
    

if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--input_catalog", default = "tests/points.dat")
    parser.add_argument("-out", "--output_catalog", default = "tests/voids_pydive.dat")
    parser.add_argument("-box", "--box_size", dest="box_size", default=2500, type=np.float64)
    parser.add_argument("-rmin", "--rmin", dest="rmin", default=0, type=np.float64)
    parser.add_argument("-rmax", "--rmax", dest="rmax", default=999, type=np.float64)
    parser.add_argument("-n", "--nthreads", dest="n_threads", default=64, type=np.int)
    parser.add_argument("-b", "--is-box", dest="is_box", action="store_true")
    parser.add_argument("-c", "--coord-conv", dest="coord_conv", action="store_true")
    parser.add_argument("-zmin", "--zmin", dest="zmin", default=0.2, type=np.float)
    parser.add_argument("-zmax", "--zmax", dest="zmax", default=0.75, type=np.float)
    parser.add_argument("-v", "--volume", dest="volume", action="store_true")
    args = parser.parse_args()
    data_fn = args.input_catalog
    is_box=args.is_box
    box_size=args.box_size
    print(f"==> Reading file {data_fn}", flush=True)
    points = pd.read_csv(data_fn, delim_whitespace=True, engine='c', names=['x', 'y', 'z'], usecols=[0,1,2]).values.astype(np.float64)
    save_galaxies_to_voids(points, args.output_catalog, r_min=args.rmin, r_max=args.rmax, box_size=box_size, is_box=is_box, cpy_range=80, n_threads=args.n_threads, coordinate_conversion=args.coord_conv, zmin=args.zmin, zmax=args.zmax, volume=args.volume)
    
    print("==> Finished successfully")
