# pydive
## Python version of [DIVE](https://github.com/cheng-zhao/DIVE)


This is the (extended) version of the DIVE code by Cheng Zhao. It is now CGAL based entirely, including features previously available only on the Scipy backend like computing simplex areas, volumes, sphericity. There are also routines available to split the void sample into central and satellite voids. 

The Scipy backend has been deprecated in the latest version in favor of the faster CGAL implementation. 

### Features
To replicate the functionality of `DIVE`, one must use the function `get_void_catalog_cgal`.
Running `pydive` on periodic boxes can now be done in two ways: With `periodic_mode=0` the boundaries of the box are extended by a distance of 5 * `(n_objects/box_volume)**(-1./3)`. This setting is much faster but uses up more memory due to the copying of points. With `periodic_mode=1` the periodic triangulation data structures in CGAL are used. For some reason this results in ~5x slower run time.

In addition, one may compute other features of the triangulation. For now, you may compute simplex area, volume, DTFE density estimation (at points and void positions). In a future a feature selection could be added to improve performance. To do this on periodic boxes, only duplicating boundaries is available given that `CGAL vertex info` is used and that is not available for periodic triangulation vertices for now. These features are available with the `get_void_catalog_full` function. See below for use examples. The output columns for the **void catalog** are 

|x|y|z|circumsphere radius|simplex volume|IDW DTFE estimate|simplex surface area|
|-|-|-|-|-|-|-|

The DTFE estimates of the point density are also returned.


I have also added routines for sky to cartesian coordinate conversion (which use GSL).

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
sys.path.append("/home/astro/dforero/codes/pydive/pydive") # add library location to path
from pydive.pydive import get_void_catalog_cgal

N = int(5e5)
np.random.seed(42)
points_raw = np.random.random((N,4)) * 2500
s = time.time()
periodic=True
voids = get_void_catalog_cgal(points_raw, 
                            periodic=periodic, 
                            periodic_mode = 0
                            )
print(f"CGAL took {time.time() - s} s", flush=True)
# Select points inside the original box
mask = (voids[:,:3] > 0).all(axis=1) & (voids[:,:3] < 2500).all(axis=1)
voids = voids[mask]

```
Computing extra features:
```python
sys.path.append("/home/astro/dforero/codes/pydive/pydive") # add library location to path
from pydive.pydive import get_void_catalog_full

N = int(5e5)
np.random.seed(42)
points_raw = np.random.random((N,4)) * 2500
s = time.time()
periodic=True
voids, dtfe = get_void_catalog_full(points_raw, 
                            periodic=periodic, 
                            )
# dtfe corresponds to the DTFE density estimation at point positions.
# so far no selection function is considered but it should be in a near future
print(f"CGAL took {time.time() - s} s", flush=True)
# Select points inside the original box
mask = (voids[:,:3] > 0).all(axis=1) & (voids[:,:3] < 2500).all(axis=1)
voids = voids[mask]

```
![alt text](https://github.com/dforero0896/pydive/blob/cgal/tests/dtfe.png?raw=true)
