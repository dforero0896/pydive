cd pydive
rm pydive.cpp
/home/astro/dforero/lib/CGAL-5.5.1/scripts/cgal_create_CMakeLists delaunay_backend
cmake -DCGAL_DIR=/global/u1/d/dforero/lib/CGAL-5.4/lib/cmake/CGAL -DCMAKE_BUILD_TYPE=Release .
make
