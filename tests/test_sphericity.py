from pydive.pydive import get_void_catalog_dtfe, get_sphericity, extend_boundaries_box, get_void_catalog
from scipy.spatial import Delaunay
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def pickle_saver(object, filename):
    print(f"==> Saving data to {filename}", flush=True)
    with open(filename+".TMP", "wb") as f:
        pickle.dump(object, f)
    os.rename(filename+".TMP", filename)


def pickle_loader(filename):
    print(f"==> Loading data from {filename}", flush=True)
    with  open(filename, "rb") as f:
        result = pickle.load(f)
    return result
if __name__ == '__main__':

    tess = pickle_loader("tests/tess.pkl")
    simplex_coords=tess.points[tess.simplices[:,:], :]
    sphericities = np.zeros((simplex_coords.shape[0],3), dtype=np.double)
    void_cat = np.zeros((simplex_coords.shape[0],4), dtype=np.double)
    n_threads = 32
    get_sphericity(simplex_coords, sphericities, sphericities.shape[0], n_threads)
    get_void_catalog(simplex_coords, void_cat, simplex_coords.shape[0])

    bins_r = np.linspace(0, 40, 100)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ax = axes.ravel()
    bins = np.logspace(np.log10(sphericities[:,0].min()), np.log10(sphericities[:,0].max()), 100)    
    ax[0].hist(sphericities[:,0], bins=bins, histtype='step')
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].set_xlabel('sphericity')

    ax[3].hist2d(void_cat[:,3], sphericities[:,0], bins=(bins_r, bins), norm=matplotlib.colors.LogNorm())
    ax[3].set_yscale('log')

    bins = np.logspace(np.log10(sphericities[:,1].min()), np.log10(sphericities[:,1].max()), 100)    
    ax[1].hist(sphericities[:,1], bins=bins, histtype='step')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('area')
    bins = np.logspace(np.log10(sphericities[:,2].min()), np.log10(sphericities[:,2].max()), 100)    
    ax[2].hist(sphericities[:,2], bins=bins, histtype='step')
    ax[2].set_yscale('log')
    ax[2].set_xscale('log')
    ax[2].set_xlabel('volume')
    
    fig.savefig("tests/sphericities.png", dpi=300)

