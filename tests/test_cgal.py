import numpy as np
import sys, time, os
sys.path.append("./pydive")
from pydive.pydive import delaunay, get_void_catalog
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    N = int(1e4)
    np.random.seed(42)
    points = np.random.random((N,3)) * 2500
    s = time.time()
    voids = delaunay(points)
    print(f"CGAL took {time.time() - s} s")
    print(voids)
    #print(pd.DataFrame(voids, columns=['x', 'y', 'z', 'r']).describe())
    plt.hist(voids[:,3], bins=100, histtype='step', label='cgal')

    s = time.time()
    tess = Delaunay(points)
    vertices = tess.points[tess.simplices[:,:], :]
    n_simplices = vertices.shape[0]
    voids = np.zeros((n_simplices, 4), dtype=np.double)
    get_void_catalog(vertices,  voids, n_simplices)
    print(f"Scipy took {time.time() - s} s")
    print(voids)
    #print(pd.DataFrame(voids, columns=['x', 'y', 'z', 'r']).describe())
    plt.hist(voids[:,3], bins=100, histtype='step', ls='--', label='scipy')

    plt.legend()

    plt.savefig("tests/distributions.png")