#!/usr/bin/env python
import sys
import time
#sys.path.append("./pydive")
from pydive.pydive import get_void_catalog_dtfe, interpolate_at_circumcenters
from dive import extend_boundaries_box
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numba
import pickle
import os

import MAS_library as MASL
import smoothing_library as SL
import Pk_library as PKL
import readfof


HALOS="/hpcstorage/zhaoc/PATCHY_BOX/pre-recon/halo/BDM_Apk/CATALPTCICz0.562G960S1010008301.dat"
DM_FIELD="/hpcstorage/zhaoc/PATCHY_BOX/pre-recon/DMfield/1010008301.dat"
BOX_SIZE=2500
GRID_SIZE=960
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
@numba.njit(numba.float64(numba.float64[:,:]))
def vol_tetrahedron(vertices):
    # Accepts points defining the simplex/tetrahedron (4, 3)
    matrix = np.empty((4, 4), dtype=np.double)
    for i in range(3):
        for j in range(4):
            elem = vertices[j,i] 
            matrix[i,j] = elem
    matrix[3,:] = 1
    #print(matrix)
    
    volume = np.abs(np.linalg.det(matrix) / 6)

    return volume
@numba.njit(parallel=False)
def get_density3d_tracer(volumes_buffer, densities_buffer, simplex_indices, simplex_coordinates, simplex_volumes, average_density = 4e-4):
    factor = 4 / average_density
    n_simplices = simplex_indices.shape[0]
    n_galaxies = volumes_buffer.shape[0]
    
    for i in numba.prange(n_simplices):
        #volume = vol_tetrahedron(simplex_coordinates[i,:,:])
        volume = simplex_volumes[i]
        assert(volume>0)
        for k in range(simplex_indices.shape[1]):
            vertex_index = simplex_indices[i,k]
            volumes_buffer[vertex_index] += volume    
    for i in numba.prange(n_galaxies):
        if volumes_buffer[i] > 0:
            densities_buffer[i] = factor / volumes_buffer[i]
    return volumes_buffer, densities_buffer


if __name__ == '__main__':

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    ax = axes.T.ravel()
    
    print(f"Loading DM field", flush=True)
    delta_dm = np.fromfile(DM_FIELD, dtype=np.float32).reshape(960, 960, 960)
    
    # %%
    


    #points_raw = pd.read_csv('tests/points.dat', engine='c', delim_whitespace=True,
    #                    names=['x', 'y', 'z'], usecols=(0, 1, 2)).values

    points_raw = pd.read_csv(HALOS, engine='c', delim_whitespace=True,
                        names=['x', 'y', 'z'], usecols=(0, 1, 2)).values.astype(np.double)
    
    points = extend_boundaries_box(points_raw, box_size=BOX_SIZE, cpy_range=40, low_range=0).astype(np.double)
    print(pd.DataFrame(points, columns=['x', 'y', 'z']).describe())
    
    ngal = points_raw.shape[0] / BOX_SIZE**3
    print(ngal, flush=True)
    tess_fn = "tess.pkl"
    if not os.path.isfile(tess_fn):
        print(f"==> Doing tesselation", flush=True)
        tess = Delaunay(points)
        pickle_saver(tess, tess_fn)
    else:
        tess = pickle_loader(tess_fn)
    print("\tDone", flush=True)
    
    # %%
    simplex_coordinates = points[tess.simplices[:,:], :]
    dummy_one = np.array([1], dtype=np.double)
    weights = np.lib.stride_tricks.as_strided(dummy_one, (points.shape[0],), (0,))
    selection = weights
    
    voids = np.zeros((simplex_coordinates.shape[0], 5), dtype=np.double)
    volumes = np.zeros((points.shape[0],), dtype=np.double)
    density = np.empty((points.shape[0],), dtype=np.double)
    density[:] = np.nan
    
    get_void_catalog_dtfe(simplex_coordinates, 
                            tess.simplices,
                            weights,
                            selection,
                            voids,
                            volumes,
                            density,
                            simplex_coordinates.shape[0], 
                            points.shape[0],
                            ngal,
                            4)
    #exit()
    
    print((volumes==0).sum(), tess.coplanar.shape, flush=True)
    vertices_domain = ((points>0) & (points<BOX_SIZE)).reshape(points.shape[0], 3).all(axis=1)
    simplices_domain = ((voids[:,:3]>0) & (voids[:,:3]<BOX_SIZE)).reshape(voids[:,:3].shape[0], 3).all(axis=1)
    finite_density_mask = volumes!=0
    density -= 1
    # %%

    
    interp = LinearNDInterpolator(tess, density)
    print(f"==> Interpolating DTFE densities at circumcenters", flush=True)
    s = time.time()
    density_at_voids_dtfe = interp(voids[:,:3])
    print(f"\t Done in {time.time() - s} s", flush=True)
    print(np.nanmin(density_at_voids_dtfe), np.nanmax(density_at_voids_dtfe), np.nanmean(density_at_voids_dtfe))
    nan_mask = np.isnan(density_at_voids_dtfe)


    print(f"==> Interpolating DTFE densities at circumcenters mine", flush=True)
    s = time.time()
    density_at_voids_dtfe_pydive = np.zeros((tess.simplices.shape[0],), dtype=np.double)
    interpolate_at_circumcenters(density,
                                 points,
                                 tess.simplices,
                                 voids[:,:4],
                                 density_at_voids_dtfe_pydive,
                                 0,
                                 16)
    print(f"\t Done in {time.time() - s} s", flush=True)
    print(np.nanmin(density_at_voids_dtfe_pydive[~nan_mask]), np.nanmax(density_at_voids_dtfe_pydive[~nan_mask]), np.nanmean(density_at_voids_dtfe_pydive[~nan_mask]))
    print("\tDone", flush=True)
    
    neighbor_volume = voids[:,4] + voids[tess.neighbors, 4].sum(axis=1)

    density_at_voids_neigbor = 4 / neighbor_volume 
    density_at_voids_neigbor /= ngal
    density_at_voids_neigbor -= 1

    # %%
    interp_volumes = NearestNDInterpolator(voids[:,:3], density_at_voids_neigbor)
    x = np.linspace(0, BOX_SIZE, GRID_SIZE)
    X, Y, Z = np.meshgrid(x, x, x)
    if not os.path.isfile("delta_neighbors_patchy.npy"):
        print(f"==> Interpolating to grid", flush=True)
        density_neighbors_grid  = interp_volumes(X.reshape(-1), Y.reshape(-1), Z.reshape(-1))
        density_neighbors_grid = np.einsum('ijk->jik', density_neighbors_grid.reshape((GRID_SIZE, GRID_SIZE, GRID_SIZE)))
        np.save("delta_neighbors_patchy.npy", density_neighbors_grid)
    else:
        print(f"==> Loading data", flush=True)
        density_neighbors_grid = np.load("delta_neighbors_patchy.npy")
    density_neighbors_grid/=np.mean(density_neighbors_grid); density_neighbors_grid -= 1
    # %%

    # density field parameters
    grid    = GRID_SIZE    #the 3D field will have grid x grid x grid voxels
    MAS     = 'CIC'  #mass-assigment scheme
    verbose = True   #print information on progress
    if not os.path.isfile("delta_cic_patchy.npy"):
        print(f"==> Interpolating to grid", flush=True)
        # define 3D density field
        density_cic_grid = np.zeros((grid,grid,grid), dtype=np.float32)

        # construct 3D density field
        MASL.MA(points_raw.astype(np.float32), density_cic_grid, BOX_SIZE, MAS, verbose=verbose)

        # at this point, delta contains the effective number of particles in each voxel
        # now compute overdensity and density constrast
        
        density_cic_grid /= np.mean(density_cic_grid, dtype=np.float64);  density_cic_grid -= 1.0
        np.save("delta_cic_patchy.npy", density_cic_grid)
    else:
        print(f"==> Loading data", flush=True)
        density_cic_grid = np.load("delta_cic_patchy.npy")
    print(np.mean(density_cic_grid))

    #R       = 5.0  #Mpc.h
    #grid    = density_cic_grid.shape[0]
    #Filter  = 'Top-Hat'
    #threads = 16

    # compute FFT of the filter
    #W_k = SL.FT_filter(BOX_SIZE, R, grid, Filter, threads)

    # smooth the field
    #field_smoothed = SL.field_smoothing(density_cic_grid.astype(np.float32), W_k, threads)
    #density_cic_grid = field_smoothed
    #density_cic_grid = delta_dm

    if not os.path.isfile("density_dtfe_patchy.npy"):
        print(f"==> Interpolating to grid", flush=True)
        density_dtfe_grid = interp(X.reshape(-1), Y.reshape(-1), Z.reshape(-1))
        density_dtfe_grid = np.einsum('ijk->jik', density_dtfe_grid.reshape((GRID_SIZE, GRID_SIZE, GRID_SIZE)))
        np.save("density_dtfe_patchy.npy", density_dtfe_grid)
    else:
        print(f"==> Loading data", flush=True)
        density_dtfe_grid = np.load("density_dtfe_patchy.npy")


    density_at_voids_cic = np.zeros(voids.shape[0], dtype=np.float32)
    MASL.CIC_interp(density_cic_grid.astype(np.float32), BOX_SIZE, voids[:,:3].astype(np.float32), density_at_voids_cic)


    voids = voids[simplices_domain]
    density_at_voids_cic = density_at_voids_cic[simplices_domain]
    density_at_voids_dtfe = density_at_voids_dtfe[simplices_domain]
    density_at_voids_dtfe_pydive = density_at_voids_dtfe_pydive[simplices_domain]
    density = density[vertices_domain]
    density_at_voids_neigbor = density_at_voids_neigbor[simplices_domain]
    

    # %%

    bins_delta = np.concatenate((-np.logspace(0.1, -5, 101), np.logspace(-5, 3, 201)))
    bins_r = np.linspace(0, 50, 101)

    cut = 2.4
    power_law = cut * (ngal * (bins_delta + 1))**(-0.24)
    '''
    ax[0].hist2d(density_at_voids_cic, voids[:,3], bins=(bins_delta, bins_r), norm=mcolors.LogNorm())
    ax[0].plot(bins_delta, power_law, c='r')
    ax[0].set_xlabel("$\delta$ CIC")
    ax[0].set_ylabel("$R$")
    ax[0].set_xscale('symlog')
    ax[0].set_yscale('symlog')
    ax[1].hist2d(density_at_voids_dtfe, voids[:,3], bins=(bins_delta, bins_r), norm=mcolors.LogNorm())
    ax[1].plot(bins_delta, power_law, c='r')
    ax[1].set_xlabel("$\delta$ DTFE")
    ax[1].set_ylabel("$R$")
    ax[1].set_xscale('symlog')
    ax[1].set_yscale('symlog')
    ax[2].hist2d(density_at_voids_neigbor, voids[:,3], bins=(bins_delta, bins_r), norm=mcolors.LogNorm())
    ax[2].plot(bins_delta, power_law, c='r')
    ax[2].set_xlabel("$\delta$ Nb")
    ax[2].set_ylabel("$R$")
    ax[2].set_xscale('symlog')
    ax[2].set_yscale('symlog')
    '''
    ax[0].hist2d(density_cic_grid.flatten() + 1, delta_dm.flatten() + 1, bins=(bins_delta, bins_delta), norm=mcolors.LogNorm())
    ax[0].set_ylabel("$\delta+1$ DM")
    ax[0].set_xlabel("$\delta+1$ CIC")
    ax[0].set_xscale('symlog')
    ax[0].set_yscale('symlog')
    ax[1].hist2d(density_dtfe_grid.flatten() + 1, delta_dm.flatten() + 1, bins=(bins_delta, bins_delta), norm=mcolors.LogNorm())
    ax[1].set_ylabel("$\delta+1$ DM")
    ax[1].set_xlabel("$\delta+1$ DTFE")
    ax[1].set_xscale('symlog')
    ax[1].set_yscale('symlog')
    ax[2].hist2d(density_neighbors_grid.flatten() + 1, delta_dm.flatten() + 1, bins=(bins_delta, bins_delta), norm=mcolors.LogNorm())
    ax[2].set_ylabel("$\delta+1$ DM")
    ax[2].set_xlabel("$\delta+1$ NB")
    ax[2].set_xscale('symlog')
    ax[2].set_yscale('symlog')



    bins = np.logspace(-3, 4, 200)
    density=True
    #ax[3].hist((1+density_at_voids_cic), bins=bins, histtype='step', label='cic cc', density=density)
    #ax[3].hist((1+density_at_voids_dtfe), bins=bins, histtype='step', label='dtfe cc', density=density)
    #ax[3].hist((1+density_at_voids_neigbor), bins=bins, histtype='step', label='Nb cc', density=density)
    #ax[3].legend()
    #ax[3].set_xlabel("$1+\delta$")
    #ax[3].set_xscale('log')
    print((1+density_cic_grid)[:,:,200:500].min(), (1+density_dtfe_grid)[:,:,200:500].min(), (1+density_neighbors_grid)[:,:,200:500].min())
    print((1+density_cic_grid)[:,:,200:500].max(), (1+density_dtfe_grid)[:,:,200:500].max(), (1+density_neighbors_grid)[:,:,200:500].max())
    vmin = 1e-1
    vmax = 1e1
    p = ax[4].imshow((1+density_cic_grid)[:,:,200:500].mean(axis=2), norm=mcolors.SymLogNorm(linthresh=1e-3))#, vmin=vmin, vmax=vmax), label = 'CIC')
    ax[4].set_title('CIC')
    fig.colorbar(p, ax = ax[4])
    ax[4].legend()
    p=ax[5].imshow((1+density_dtfe_grid)[:,:,200:500].mean(axis=2), norm=mcolors.SymLogNorm(linthresh=1e-3))#, vmin=vmin, vmax=vmax), label = 'DTFE')
    ax[5].set_title('DTFE')
    fig.colorbar(p, ax = ax[5])
    ax[5].legend()
    p=ax[6].imshow((1+density_neighbors_grid)[:,:,200:500].mean(axis=2), norm=mcolors.SymLogNorm(linthresh=1e-3))#, vmin=vmin, vmax=vmax), label = 'Nb')
    ax[6].set_title('Nb')
    fig.colorbar(p, ax = ax[6])
    ax[6].legend()
    p=ax[3].imshow((1+delta_dm)[:,:,200:500].mean(axis=2), norm=mcolors.SymLogNorm(linthresh=1e-3))#, vmin=vmin, vmax=vmax), label = 'DM')
    ax[3].set_title('DM')
    fig.colorbar(p, ax = ax[3])
    ax[3].legend()
    

    # Correlations

    MAS     = 'CIC'
    axis    = 0
    threads = 16

    # compute the correlation function
    CF_dm     = PKL.Xi(delta_dm.astype(np.float32), BOX_SIZE, 'None', axis, threads)
    dm_mask = CF_dm.r3D < 200
    CF_cic     = PKL.Xi(density_cic_grid.astype(np.float32), BOX_SIZE, MAS, axis, threads)
    cic_mask = CF_cic.r3D < 200
    CF_dtfe     = PKL.Xi(density_dtfe_grid.astype(np.float32), BOX_SIZE, 'None', axis, threads)
    dtfe_mask = CF_dtfe.r3D < 200
    CF_nb     = PKL.Xi(density_neighbors_grid.astype(np.float32), BOX_SIZE, 'None', axis, threads)
    nb_mask = CF_nb.r3D < 200
    XCF = PKL.XXi(density_cic_grid.astype(np.float32), density_dtfe_grid.astype(np.float32), BOX_SIZE, ['CIC', 'None'], axis, threads)
    cross_mask = XCF.r3D < 200
    XCF_dm = PKL.XXi(delta_dm.astype(np.float32), density_dtfe_grid.astype(np.float32), BOX_SIZE, ['None', 'None'], axis, threads)
    crossdm_mask = XCF_dm.r3D < 200

    ax[7].plot(CF_dm.r3D[dm_mask], (CF_dm.r3D**2*CF_dm.xi[:,0])[dm_mask], label='DM', ls='-', c='k')
    ax[7].plot(CF_cic.r3D[cic_mask], (CF_cic.r3D**2*CF_cic.xi[:,0])[cic_mask], label='CIC', ls='-')
    ax[7].plot(CF_dtfe.r3D[dtfe_mask], (CF_dtfe.r3D**2*CF_dtfe.xi[:,0])[dtfe_mask], label='DTFE', ls='--')
    ax[7].plot(CF_nb.r3D[nb_mask], (CF_nb.r3D**2*CF_nb.xi[:,0])[nb_mask], label='Nb', ls='--')
    ax[7].plot(XCF.r3D[cross_mask], (XCF.r3D**2*XCF.xi[:,0])[cross_mask], label='Cross CIC', ls=':')
    ax[7].plot(XCF_dm.r3D[crossdm_mask], (XCF_dm.r3D**2*XCF_dm.xi[:,0])[crossdm_mask], label='Cross DM', ls=':')
    ax[7].legend(loc=0)


    Pk_cic = PKL.Pk(density_cic_grid.astype(np.float32), BOX_SIZE, axis, 'CIC', threads)
    Pk_dtfe = PKL.Pk(density_dtfe_grid.astype(np.float32), BOX_SIZE, axis, 'None', threads)






    fig.tight_layout()

    fig.savefig('dtfe.png', dpi=200)

    


    #ax[7].plot(Pk_cic.k1D, Pk_cic.Pk1D, label='CIC')
    #ax[7].plot(Pk_dtfe.k1D, Pk_dtfe.Pk1D, label='DTFE')
    #ax[7].set_xscale('symlog')
    #ax[7].set_yscale('symlog')
    







