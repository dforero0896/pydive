# pydive
## Python version of [DIVE](https://github.com/cheng-zhao/DIVE)


This is the (extended) version of the DIVE code by Cheng Zhao. It offers two interchangeable backends:

- **CGAL backend** (`backend='cgal'`): the fastest implementation, entirely CGAL based, including features previously available only on the SciPy backend like computing simplex areas, volumes, sphericity. There are also routines available to split the void sample into central and satellite voids, and sky-to-cartesian coordinate conversions (which use GSL). Requires compiling against CGAL/GSL (see Compilation notes below and `INSTALL_CGAL.md`).
- **SciPy + Numba backend** (`backend='scipy'`): a pure-Python fallback using `scipy.spatial.Delaunay` (Qhull) with JIT-compiled (Numba) loops for all heavy computation — no Python-level loops over simplices. Easier to install (no C++ toolchain needed); somewhat slower than CGAL but fully functional, including periodic boundaries.

The top-level API auto-selects the best available backend:

```python
from pydive import get_void_catalog, get_void_catalog_full, check_backend_status
voids = get_void_catalog(points)                  # CGAL if available, else SciPy
voids = get_void_catalog(points, backend='scipy') # force SciPy
check_backend_status()                            # show what is installed
```

### Features
To replicate the functionality of `DIVE`, one must use the function `get_void_catalog_cgal`.
Running `pydive` on periodic boxes can be done in two ways with the CGAL backend: With `periodic_mode=0` the boundary shells of the box are replicated out to a distance `cpy_range` (default: 8 * `(n_objects/box_volume)**(-1./3)`, i.e. 8 times the mean interparticle distance; only points within that range of each face are copied, not the full box images). This setting is much faster but uses up more memory due to the copying of points. With `periodic_mode=1` the true periodic triangulation data structures in CGAL are used (no duplication at all). For some reason this results in ~5x slower run time.

The SciPy backend uses the same shell-replication strategy (`mode='periodic'`, `boxsize=...`, optional `cpy_range=...`) because Qhull (`scipy.spatial.Delaunay`) has **no native periodic Delaunay triangulation**; the padding is therefore a requirement of SciPy, not just a workaround for CGAL limitations. The default copy range matches the CGAL backend so both produce consistent catalogs.

Boundary modes for the SciPy backend: `'open'` (no padding), `'periodic'` (replicate boundary shells), and `'lightcone'` — which is treated exactly like `'open'`, since lightcones are not periodic and must not be wrapped.

In addition, one may compute other features of the triangulation. For now, you may compute simplex area, volume, DTFE density estimation (at points and void positions). In a future a feature selection could be added to improve performance. To do this on periodic boxes, only duplicating boundaries is available given that CGAL vertex info is used and that is not available for periodic triangulation vertices for now. These features are available with the `get_void_catalog_full` function. See below for use examples.

## Compilation notes

Some parts of the code need GSL so make sure to link to it. If GSL is installed in your system wou may use `gsl-config` to find out the paths to the libraries and headers. CGAL requires the codes using it to use `cmake` too. The script `run_cmake.sh` handles the cmake part of the build, you should only make sure that the path to the  `cgal_create_CMakeLists` is properly set, it should be in your CGAL installation directory, i.e. `PATH/TO/CGAL/CGAL-5.4/bin/cgal_create_CMakeLists`. Once you have set the path, you can then make sure that the `include` and `library` paths in `setup.py` are correctly set. Once this is done, the code can be built with the `make` command. You may find more informtion on building software with CGAL [here](https://doc.cgal.org/latest/Manual/installation.html). To use the code you must add the directory `pydive` to your python path usign e.g. `export PYTHONPATH=/path/to/pydive/pydive:$PYTHONPATH` or using `sys.path.append()`.


For information about the motivation, references and original implementation, please visit [DIVE's repository](https://github.com/cheng-zhao/DIVE). 

If you use this implementation in a scientific publication, please link to this repository and cite the following papers
```
@ARTICLE{2021arXiv210702950F,
       author = {{Forero-S{\'a}nchez}, Daniel and {Zhao}, Cheng and {Tao}, Charling and {Chuang}, Chia-Hsun and {Kitaura}, Francisco-Shu and {Variu}, Andrei and {Tamone}, Am{\'e}lie and {Kneib}, Jean-Paul},
        title = "{Cosmic Void Baryon Acoustic Oscillation Measurement: Evaluation of Sensitivity to Selection Effects}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2021,
        month = jul,
          eid = {arXiv:2107.02950},
        pages = {arXiv:2107.02950},
archivePrefix = {arXiv},
       eprint = {2107.02950},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210702950F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
```
@ARTICLE{2016MNRAS.459.2670Z,
       author = {{Zhao}, Cheng and {Tao}, Charling and {Liang}, Yu and {Kitaura}, Francisco-Shu and {Chuang}, Chia-Hsun},
        title = "{DIVE in the cosmic web: voids with Delaunay triangulation from discrete matter tracer distributions}",
      journal = {\mnras},
     keywords = {methods: data analysis, catalogues, galaxies: structure, large-scale structure of Universe, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2016,
        month = jul,
       volume = {459},
       number = {3},
        pages = {2670-2680},
          doi = {10.1093/mnras/stw660},
archivePrefix = {arXiv},
       eprint = {1511.04299},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2016MNRAS.459.2670Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```

### Usage examples

`DIVE` functionality: Delaunay tesselation and computing circumsphere positions/radii.
```python
import numpy as np, time
from pydive import get_void_catalog

N = int(5e5)
np.random.seed(42)
points_raw = np.random.random((N, 3)) * 2500

# CGAL backend (requires compiled extension)
s = time.time()
voids = get_void_catalog(points_raw, backend='cgal',
                         periodic=True,
                         box_min=[0., 0., 0.], box_max=[2500.]*3,
                         periodic_mode=0)
print(f"CGAL took {time.time() - s} s", flush=True)
# With periodic_mode=0 voids outside the original box are filtered out by the
# backend; with mode selection left to the caller:
mask = (voids[:, :3] >= 0).all(axis=1) & (voids[:, :3] <= 2500).all(axis=1)
voids = voids[mask]
```

The same catalog with the SciPy+Numba backend (no compilation needed; periodic
boundaries handled via boundary-shell replication):
```python
from pydive import get_void_catalog

voids = get_void_catalog(points_raw, backend='scipy',
                         mode='periodic', boxsize=2500.)
```

Computing extra features:
```python
import numpy as np, time
from pydive import get_void_catalog_full

N = int(5e5)
np.random.seed(42)
points_raw = np.random.random((N, 3)) * 2500
s = time.time()
voids, dtfe = get_void_catalog_full(points_raw, backend='cgal',
                                    periodic=True,
                                    box_min=[0., 0., 0.], box_max=[2500.]*3)
# dtfe corresponds to the DTFE density estimation at point positions.
# so far no selection function is considered but it should be in a near future
print(f"CGAL took {time.time() - s} s", flush=True)
# Select points inside the original box
mask = (voids[:, :3] > 0).all(axis=1) & (voids[:, :3] < 2500).all(axis=1)
voids = voids[mask]
```

Or with the SciPy backend (output columns: x, y, z, r, volume, dtfe_interp, area;
the periodic filter to the original box is applied automatically):
```python
voids, dtfe = get_void_catalog_full(points_raw, backend='scipy',
                                    mode='periodic', boxsize=2500.)
```

Note on boundary modes: `'lightcone'` is equivalent to `'open'` — lightcones
are not periodic, so no wrapping/padding is applied in that direction.
![alt text](https://github.com/dforero0896/pydive/blob/cgal/tests/dtfe.png?raw=true)
