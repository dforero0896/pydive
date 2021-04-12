#!/usr/bin/env python
import sys

sys.path.append("./pydive")
from pydive.pydive import get_void_catalog_wdensity
from dive import extend_boundaries_box
from scipy.spatial import Delaunay
import pandas as pd
import numpy as np



if __name__ == '__main__':
    
    points = pd.read_csv('tests/points.dat', engine='c', delim_whitespace=True,
                        names=['x', 'y', 'z'], usecols=(0, 1, 2)).values
    points = extend_boundaries_box(points, box_size=2500, cpy_range=80, low_range=0)
    tess = Delaunay(points)

    simplex_coordinates = tess.points[tess.simplices[:,:], :]
    dummy_one = np.array([1], dtype=np.double)
    weights = np.lib.stride_tricks.as_strided(dummy_one, (points.shape[0],), (0,))
    selection = weights

    voids = np.zeros((simplex_coordinates.shape[0], 5), dtype=np.double)
    volumes = np.zeros((points.shape[0],))
    density = np.zeros((points.shape[0],))
    
    get_void_catalog_wdensity(simplex_coordinates, 
                                tess.simplices,
                                weights,
                                selection,
                                voids,
                                volumes,
                                density,
                                simplex_coordinates.shape[0], 
                                points.shape[0], 
                                4)
    print(density[density!=-99999])