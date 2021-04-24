#!/usr/bin/env python
import sys
import time
sys.path.append("./pydive")
from pydive.pydive import get_void_catalog_wdensity, interpolate_at_circumcenters
from dive import extend_boundaries_box
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numba

import MAS_library as MASL
import smoothing_library as SL
import Pk_library as PKL
import readfof



QUIJOTE_SNAPDIR="/global/cscratch1/sd/zhaoc/Quijote/Halos/fiducial/0/"
QUIJOTE_SNAPNUM=4 # z=0
z_dict = {4:0.0, 3:0.5, 2:1.0, 1:2.0, 0:3.0}
redshift = z_dict[QUIJOTE_SNAPNUM]

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
    
    


    #points_raw = pd.read_csv('tests/points.dat', engine='c', delim_whitespace=True,
    #                    names=['x', 'y', 'z'], usecols=(0, 1, 2)).values

    FoF = readfof.FoF_catalog(QUIJOTE_SNAPDIR, QUIJOTE_SNAPNUM, long_ids=False,
                          swap=False, SFR=False, read_IDs=False)

# get the properties of the halos
    points_raw = (FoF.GroupPos/1e3).astype(np.double)            #Halo positions in Mpc/h
    
    points = extend_boundaries_box(points_raw, box_size=1000, cpy_range=80, low_range=0).astype(np.double)
    print(pd.DataFrame(points, columns=['x', 'y', 'z']).describe())
    
    ngal = points_raw.shape[0] / 1000**3
    print(ngal, flush=True)
    print(f"==> Doing tesselation", flush=True)
    #tess = Delaunay(points - points.mean(axis=0), qhull_options = "QJ")
    #points=points_raw
    tess = Delaunay(points)
    print("\tDone", flush=True)
    print(tess.points[[774795,                                                                                                                                                                                        
                        618318,                                                                                                                                                      
                        431163,                                                                                                                                                                                          
                        771415]])
    #exit()
    simplex_coordinates = points[tess.simplices[:,:], :]
    dummy_one = np.array([1], dtype=np.double)
    weights = np.lib.stride_tricks.as_strided(dummy_one, (points.shape[0],), (0,))
    selection = weights
    
    voids = np.zeros((simplex_coordinates.shape[0], 5), dtype=np.double)
    volumes = np.zeros((points.shape[0],), dtype=np.double)
    density = np.empty((points.shape[0],), dtype=np.double)
    density[:] = np.nan
    
    get_void_catalog_wdensity(simplex_coordinates, 
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
    vertices_domain = ((points>0) & (points<1000)).reshape(points.shape[0], 3).all(axis=1)
    simplices_domain = ((voids[:,:3]>0) & (voids[:,:3]<1000)).reshape(voids[:,:3].shape[0], 3).all(axis=1)
    finite_density_mask = volumes!=0
    print(np.nanmean(density[vertices_domain & finite_density_mask]))
    print(volumes[volumes!=0])
    print(density[volumes!=0])
    print(density[volumes==0])
    density -= 1
    print(np.nanmean(density[vertices_domain & finite_density_mask]))

    
    interp = LinearNDInterpolator(points, density)
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
    


    # density field parameters
    grid    = 256    #the 3D field will have grid x grid x grid voxels
    BoxSize = 1000.0 #Mpc/h ; size of box
    MAS     = 'CIC'  #mass-assigment scheme
    verbose = True   #print information on progress

    # define 3D density field
    density_cic_grid = np.zeros((grid,grid,grid), dtype=np.float32)

    # construct 3D density field
    MASL.MA(points_raw.astype(np.float32), density_cic_grid, BoxSize, MAS, verbose=verbose)

    # at this point, delta contains the effective number of particles in each voxel
    # now compute overdensity and density constrast
    density_cic_grid /= np.mean(density_cic_grid, dtype=np.float64);  density_cic_grid -= 1.0
    
    print(np.mean(density_cic_grid))

    BoxSize = 1000
    R       = 15.0  #Mpc.h
    grid    = density_cic_grid.shape[0]
    Filter  = 'Top-Hat'
    threads = 16

    # compute FFT of the filter
    W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)

    # smooth the field
    field_smoothed = SL.field_smoothing(density_cic_grid.astype(np.float32), W_k, threads)
    density_cic_grid = field_smoothed

    x = np.linspace(0, 1000, 256)
    X, Y, Z = np.meshgrid(x, x, x)

    density_dtfe_grid = interp(X.reshape(-1), Y.reshape(-1), Z.reshape(-1))
    density_dtfe_grid = np.einsum('ijk->jik', density_dtfe_grid.reshape((256, 256, 256)))


    density_at_voids_cic = np.zeros(voids.shape[0], dtype=np.float32)
    MASL.CIC_interp(density_cic_grid.astype(np.float32), BoxSize, voids[:,:3].astype(np.float32), density_at_voids_cic)


    voids = voids[simplices_domain]
    density_at_voids_cic = density_at_voids_cic[simplices_domain]
    density_at_voids_dtfe = density_at_voids_dtfe[simplices_domain]
    density_at_voids_dtfe_pydive = density_at_voids_dtfe_pydive[simplices_domain]
    density = density[vertices_domain]
    bins_delta = np.concatenate((-np.logspace(0.1, -5, 101), np.logspace(-5, 3, 201)))
    bins_r = np.linspace(0, 50, 101)

    ax[0].hist2d(density_at_voids_cic, voids[:,3], bins=(bins_delta, bins_r), norm=mcolors.LogNorm())
    ax[0].set_xlabel("$\delta$ CIC")
    ax[0].set_ylabel("$R$")
    ax[0].set_xscale('symlog')
    ax[1].hist2d(density_at_voids_dtfe, voids[:,3], bins=(bins_delta, bins_r), norm=mcolors.LogNorm())
    ax[1].set_xlabel("$\delta$ DTFE")
    ax[1].set_ylabel("$R$")
    ax[1].set_xscale('symlog')
    ax[2].hist2d(density_at_voids_dtfe_pydive, voids[:,3], bins=(bins_delta, bins_r), norm=mcolors.LogNorm())
    ax[2].set_xlabel("$\delta$ DTFE pydive")
    ax[2].set_ylabel("$R$")
    ax[2].set_xscale('symlog')

    bins = np.logspace(-7, np.log10(100.), 100)
    ax[3].hist((1+density_cic_grid).flatten(), bins=bins, histtype='step', label='cic')
    ax[3].hist((1+density_dtfe_grid).flatten(), bins=bins, histtype='step', label='dtfe')
    ax[3].legend()
    ax[3].set_xlabel("$1+\delta$")
    ax[3].set_xscale('log')

    p = ax[4].imshow((1+density_cic_grid)[:,:,50:70].mean(axis=2), norm=mcolors.SymLogNorm(linthresh=1e-3, vmin=1e-1, vmax=0.7e1), label = 'CIC')
    fig.colorbar(p, ax = ax[4])
    ax[4].legend()
    p=ax[5].imshow((1+density_dtfe_grid)[:,:,50:70].mean(axis=2), norm=mcolors.SymLogNorm(linthresh=1e-3, vmin=1e-1, vmax=0.7e1), label = 'DTFE')
    fig.colorbar(p, ax = ax[5])
    ax[5].legend()


    BoxSize = 1000.0 #Mpc/h
    MAS     = 'CIC'
    axis    = 0
    threads = 16

    # compute the correlation function
    CF_cic     = PKL.Xi(density_cic_grid.astype(np.float32), BoxSize, MAS, axis, threads)
    cic_mask = CF_cic.r3D < 200
    CF_dtfe     = PKL.Xi(density_dtfe_grid.astype(np.float32), BoxSize, 'None', axis, threads)
    dtfe_mask = CF_dtfe.r3D < 200
    XCF = PKL.XXi(density_cic_grid.astype(np.float32), density_dtfe_grid.astype(np.float32), BoxSize, ['CIC', 'None'], axis, threads)
    cross_mask = XCF.r3D < 200


    ax[6].plot(CF_cic.r3D[cic_mask], (CF_cic.r3D**2*CF_cic.xi[:,0])[cic_mask], label='CIC', ls='-')
    ax[6].plot(CF_dtfe.r3D[dtfe_mask], (CF_dtfe.r3D**2*CF_dtfe.xi[:,0])[dtfe_mask], label='DTFE', ls='--')
    ax[6].plot(XCF.r3D[cross_mask], (XCF.r3D**2*XCF.xi[:,0])[cross_mask], label='Cross', ls=':')
    ax[6].legend()


    Pk_cic = PKL.Pk(density_cic_grid.astype(np.float32), BoxSize, axis, 'CIC', threads)
    Pk_dtfe = PKL.Pk(density_dtfe_grid.astype(np.float32), BoxSize, axis, 'None', threads)

    #ax[7].plot(Pk_cic.k1D, Pk_cic.Pk1D, label='CIC')
    #ax[7].plot(Pk_dtfe.k1D, Pk_dtfe.Pk1D, label='DTFE')
    #ax[7].set_xscale('symlog')
    #ax[7].set_yscale('symlog')
    





    fig.savefig('tests/dtfe.png', dpi=200)


