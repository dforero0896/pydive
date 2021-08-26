import numpy as np
import sys, time, os
sys.path.append("/home/astro/dforero/codes/pydive/pydive")
from pydive.pydive import get_void_catalog_cgal, get_void_catalog_full
from scipy.spatial import Delaunay
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
# make sure to add CGAL to LD_LIBRARY_PATH
if __name__ == '__main__':
    N = int(5e5)
    np.random.seed(42)
    points_raw = np.random.random((N,4)) * 2500
    HALOS="/hpcstorage/zhaoc/PATCHY_BOX/pre-recon/halo/BDM_Apk/CATALPTCICz0.562G960S1010008301.dat"
    #points_raw = pd.read_csv(HALOS, engine='c', delim_whitespace=True,
    #                    names=['x', 'y', 'z'], usecols=(0, 1, 2)).values.astype(np.double)[:]
    points_raw = np.c_[points_raw, np.zeros(points_raw.shape[0])]
    buffer_ones = np.ones_like(points_raw[:,0])
    s = time.time()
    periodic=False
    fig, ax = plt.subplots(2, 2, figsize=(10,10))


    
    for periodic in [False, True]:
        
        s = time.time()
        voids = get_void_catalog_cgal(points_raw, 
                                    periodic=periodic, 
                                    )
        print(f"CGAL took {time.time() - s} s")
        mask = (voids[:,:3] > 0).all(axis=1) & (voids[:,:3] < 2500).all(axis=1)
        voids = voids[mask]
        
        voids_df = pd.DataFrame(voids, columns=['x', 'y', 'z', 'r'])
        print(voids_df.describe())

        #bins = np.logspace(-4, 11, 101)
        bins=100

        axr = ax.ravel()
        for i in range(voids.shape[1]):
            axr[i].hist(voids[:,i], histtype="step", label=f"PyDIVE CGAL {periodic}", bins=bins)
            axr[i].set_title(voids_df.columns[i])
            #axr[i].set_xscale('log')
            axr[i].legend()

        

        plt.savefig("tests/distributions.png")
