import numpy as np
import sys, time, os
sys.path.append("/home/astro/dforero/codes/pydive/pydive")
from pydive.pydive import get_void_catalog_cgal, get_void_catalog, extend_boundaries_box
from scipy.spatial import Delaunay
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
# make sure to add CGAL to LD_LIBRARY_PATH
if __name__ == '__main__':
    N = int(1e5)
    np.random.seed(42)
    points_raw = np.random.random((N,4)) * 2500
    buffer_ones = np.ones_like(points_raw[:,0])
    s = time.time()
    periodic=True
    compute_dtfe = False
    if periodic:
        points = extend_boundaries_box(points_raw, box_size=2500, cpy_range=100).astype(np.double)
    else:
        points = points_raw
    print(f"Duplicating boundaries took {time.time() - s} s")
    s = time.time()
    voids = get_void_catalog_cgal(points_raw, 
                                periodic=periodic, 
                                box_size=2500, 
                                cpy_range=100, 
                                compute_dtfe=compute_dtfe,
                                weights = buffer_ones,
                                selection = buffer_ones,
                                average_density = points_raw.shape[0] / 2500**3)
    mask = (voids[:,:3] > 0).all(axis=1) & (voids[:,:3] < 2500).all(axis=1)
    voids = voids[mask]
    print(f"CGAL took {time.time() - s} s")
    voids_df = pd.DataFrame(voids, columns=['x', 'y', 'z', 'r'])
    print(voids_df.describe())

    #bins = np.logspace(-4, 11, 101)
    bins=100
    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    axr = ax.ravel()
    for i in range(voids.shape[1]):
        axr[i].hist(voids[:,i], histtype="step", color="b", label="PyDIVE CGAL", bins=bins)
        axr[i].set_title(voids_df.columns[i])
        #axr[i].set_xscale('log')
        axr[i].legend()

    s = time.time()
    tess = Delaunay(points[:,:3])
    vertices = tess.points[tess.simplices[:,:], :]
    n_simplices = vertices.shape[0]
    voids = np.zeros((n_simplices, 4), dtype=np.double)
    get_void_catalog(vertices,  voids, n_simplices)
    mask = (voids[:,:3] > 0).all(axis=1) & (voids[:,:3] < 2500).all(axis=1)
    voids = voids[mask]
    print(f"Scipy took {time.time() - s} s")
    voids_df = pd.DataFrame(voids, columns=['x', 'y', 'z', 'r'])
    print(voids_df.describe())
    

    for i in range(voids.shape[1]):
        axr[i].hist(voids[:,i], histtype="step", color="r", label="PyDIVE Scipy", bins=bins, ls='--')
        axr[i].set_title(voids_df.columns[i])
        #axr[i].set_xscale('log')
        axr[i].legend()

    plt.legend()

    plt.savefig("tests/distributions.png")