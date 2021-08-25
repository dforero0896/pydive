#!/usr/bin/env python
import sys
import time
#sys.path.append("./pydive")
sys.path.append("/home/astro/dforero/codes/pydive/pydive")
from pydive.pydive import get_void_catalog_cgal, get_void_catalog_full
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import os

import MAS_library as MASL
import smoothing_library as SL
import Pk_library as PKL
import readfof


HALOS="/hpcstorage/zhaoc/PATCHY_BOX/pre-recon/halo/BDM_Apk/CATALPTCICz0.562G960S1010008301.dat"
DM_FIELD="/hpcstorage/zhaoc/PATCHY_BOX/pre-recon/DMfield/1010008301.dat"
BOX_SIZE=1000
GRID_SIZE=512

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

# make sure to add CGAL to LD_LIBRARY_PATH
if __name__ == '__main__':
    N = int(0.5e5)
    np.random.seed(42)
    points_raw = np.random.random((N,4)) * BOX_SIZE
    buffer_ones = np.ones_like(points_raw[:,0])
    s = time.time()
    periodic=False
    


    s = time.time()
    voids, dtfe = get_void_catalog_full(points_raw, 
                                periodic=False, 
                                )
    print(f"CGAL took {time.time() - s} s")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    ax = axes.T.ravel()
    

    x = np.linspace(0, BOX_SIZE, GRID_SIZE)
    X, Y, Z = np.meshgrid(x, x, x)
    
    #interp = LinearNDInterpolator(points_raw[:,:3], dtfe, fill_value=0)
    interp = NearestNDInterpolator(points_raw[:,:3], dtfe)
    #points = np.concatenate((points_raw[:,:3], voids[:,:3]), axis = 0)
    #values = np.concatenate((dtfe.squeeze(), voids[:,5].squeeze()))
    #interp = NearestNDInterpolator(points, values)
    density_dtfe_linterp = interp(X.reshape(-1), Y.reshape(-1), Z.reshape(-1))
    density_dtfe_linterp = np.einsum('ijk->jik', density_dtfe_linterp.reshape((GRID_SIZE, GRID_SIZE, GRID_SIZE)))
    density_dtfe_linterp /= np.mean(density_dtfe_linterp, dtype=np.float64);  density_dtfe_linterp -= 1.0


    grid    = GRID_SIZE    #the 3D field will have grid x grid x grid voxels
    MAS     = 'CIC'  #mass-assigment scheme
    R       = 5 #Mpc.h
    verbose = True   #print information on progress
    Filter  = 'Gaussian'
    if not os.path.isfile("delta_cic_patchy.npy"):
        print(f"==> Interpolating to grid", flush=True)
        # define 3D density field
        density_cic_grid = np.zeros((grid,grid,grid), dtype=np.float32)

        # construct 3D density field
        MASL.MA(points_raw[:,:3].astype(np.float32), density_cic_grid, BOX_SIZE, MAS, verbose=verbose)

        # at this point, delta contains the effective number of particles in each voxel
        # now compute overdensity and density constrast
        
        density_cic_grid /= np.mean(density_cic_grid, dtype=np.float64);  density_cic_grid -= 1.0
        W_k = SL.FT_filter(BOX_SIZE, R, grid, Filter, 8)
        density_cic_grid = SL.field_smoothing(density_cic_grid, W_k, 8)

        vmin, vmax = np.min(density_cic_grid), np.max(density_cic_grid)
    

    p = ax[4].imshow((1+density_cic_grid)[:,:,200:500].mean(axis=2), vmin=0.5, vmax=1.5)#, norm=mcolors.SymLogNorm(linthresh=1e-3))#, vmin=vmin, vmax=vmax), label = 'CIC')
    ax[4].set_title('CIC')
    fig.colorbar(p, ax = ax[4])
    ax[4].legend()
    p=ax[5].imshow((1+density_dtfe_linterp)[:,:,200:500].mean(axis=2), vmin=0.5, vmax=1.5)#, norm=mcolors.SymLogNorm(linthresh=1e-3))#, vmin=vmin, vmax=vmax), label = 'DTFE')
    ax[5].set_title('DTFE')
    fig.colorbar(p, ax = ax[5])
    ax[5].legend()

    fig.tight_layout()

    fig.savefig('tests/dtfe.png', dpi=200)
    
    # Correlations

    MAS     = 'CIC'
    axis    = 0
    threads = 2

    # compute the correlation function
    
    CF_cic     = PKL.Xi(density_cic_grid.astype(np.float32), BOX_SIZE, MAS, axis, threads)
    cic_mask = CF_cic.r3D < 200
    CF_dtfe     = PKL.Xi(density_dtfe_linterp.astype(np.float32), BOX_SIZE, 'None', axis, threads)
    dtfe_mask = CF_dtfe.r3D < 200
    XCF = PKL.XXi(density_cic_grid.astype(np.float32), density_dtfe_linterp.astype(np.float32), BOX_SIZE, [MAS, 'None'], axis, threads)
    cross_mask = XCF.r3D < 200
    

    
    ax[7].plot(CF_cic.r3D[cic_mask], (CF_cic.r3D**2*CF_cic.xi[:,0])[cic_mask], label='CIC', ls='-')
    ax[7].plot(CF_dtfe.r3D[dtfe_mask], (CF_dtfe.r3D**2*CF_dtfe.xi[:,0])[dtfe_mask], label='DTFE', ls='--')
    
    ax[7].plot(XCF.r3D[cross_mask], (XCF.r3D**2*XCF.xi[:,0])[cross_mask], label='Cross CIC', ls=':')
    
    ax[7].legend(loc=0)

        

    fig.tight_layout()

    fig.savefig('tests/dtfe.png', dpi=200)

    mask = (voids[:,:3] > 0).all(axis=1) & (voids[:,:3] < BOX_SIZE).all(axis=1)
    voids = voids[mask]

    ax[0].hist(voids[:,0], bins=100, histtype='step', label='x')
    ax[0].hist(voids[:,1], bins=100, histtype='step', label='y')
    ax[0].hist(voids[:,2], bins=100, histtype='step', label='z')
    ax[0].legend()

    ax[1].hist(voids[:,3], bins=100, histtype='step', label='r')
    ax[1].legend()

    ax[2].hist(voids[:,4], bins=100, histtype='step', label='vol')
    ax[2].legend()

    ax[3].hist(voids[:,6], bins=100, histtype='step', label='area')
    ax[3].legend()

    sphericity = (36 * np.pi * voids[:,4]**2)**(1./3) / voids[:,6]

    ax[6].hist(sphericity, bins=100, histtype='step', label='sphericity')
    ax[6].legend()

    fig.savefig('tests/dtfe.png', dpi=200)