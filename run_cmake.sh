cd pydive
rm pydive.cpp
#/home/astro/dforero/lib/CGAL-5.5.1/scripts/cgal_create_CMakeLists -s delaunay_backend
cmake -DCGAL_DIR=/home/astro/dforero/lib/CGAL-5.5.1/ -DCMAKE_BUILD_TYPE=Release -DBoost_INCLUDE_DIR=/opt/ebsofts/Boost/1.79.0-GCC-11.3.0/include .
make
