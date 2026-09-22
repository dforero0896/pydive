/**
 * Standalone CGAL Delaunay backend for pydive
 * 
 * This file provides a portable CGAL implementation that can be compiled
 * with standard Python extension building tools (setuptools/distutils).
 * 
 * The key to portability is:
 * 1. Using CGAL's header-only mode where possible
 * 2. Properly detecting CGAL installation via pkg-config or CMake
 * 3. Providing clear error messages when CGAL is not available
 */

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_traits_3.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Sphere_3.h>
#include <CGAL/Tetrahedron_3.h>
#include <CGAL/Triangle_3.h>

#include <vector>
#include <set>
#include <cassert>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iterator>
#include <iostream>

#define DOUBLE_MAX std::numeric_limits<double>::max()
#define DOUBLE_MIN std::numeric_limits<double>::min()

// Traits and triangulation data structures
typedef CGAL::Exact_predicates_exact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_3<size_t,K> Vertex_base_info;
typedef CGAL::Triangulation_data_structure_3<Vertex_base_info, CGAL::Delaunay_triangulation_cell_base_3<K>> TriangulationDS;
typedef CGAL::Periodic_3_Delaunay_triangulation_traits_3<K> P3Traits;
typedef CGAL::Periodic_3_Delaunay_triangulation_3<P3Traits> PDelaunay;
typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
typedef CGAL::Delaunay_triangulation_3<K,TriangulationDS> DelaunayInfo;

// Iterators
typedef PDelaunay::Periodic_tetrahedron_iterator periodic_tetrahedra;
typedef Delaunay::Finite_cells_iterator finite_cells;
typedef Delaunay::Finite_vertices_iterator delaunay_vertices;
typedef Delaunay::Vertex_handle vertex_handle;
typedef Delaunay::Cell_handle cell_handle;
typedef DelaunayInfo::Finite_cells_iterator finite_cells_info;

// Geometric object types
typedef PDelaunay::Iso_cuboid Iso_cuboid;
typedef CGAL::Point_3<K> Point_3;
typedef CGAL::Sphere_3<K> Sphere_3;
typedef CGAL::Tetrahedron_3<K> Tetrahedron_3;
typedef CGAL::Triangle_3<K> Triangle_3;
typedef Delaunay::Point Point;
typedef PDelaunay::Point PPoint;
typedef K::FT FT;

struct DelaunayOutput {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> r;
    std::vector<double> volume;
    std::vector<double> area;
    std::vector<double> dtfe;
    std::vector<double> weights;
    std::vector<double> selection;
    std::vector<size_t> vertices[4];
    size_t n_simplices;
};

inline double tetrahedron_area(const Tetrahedron_3& tetrahedron) {
    double tot_area = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = i+1; j < 4; j++) {
            for (int k = j+1; k < 4; k++) {
                Triangle_3 facet = Triangle_3(tetrahedron[i], tetrahedron[j], tetrahedron[k]);
                tot_area += CGAL::sqrt(CGAL::to_double(facet.squared_area()));
            }
        }
    }
    return tot_area;
}

// Basic non-periodic Delaunay triangulation
extern "C" DelaunayOutput* cdelaunay_basic(double* X, double* Y, double* Z, size_t n_points) {
    DelaunayOutput* output = new DelaunayOutput();
    std::vector<Point_3> points;
    points.reserve(n_points);

    for (size_t i = 0; i < n_points; i++) {
        points.push_back(Point_3(X[i], Y[i], Z[i]));
    }
    
    std::cout << "==> Number of points: " << points.size() << std::endl;
    std::cout << "==> Building Delaunay Triangulation." << std::endl;
    
    Delaunay tess(points.begin(), points.end());
    assert(tess.is_valid());
    points.clear();
    
    Point_3 simplex_vertices[4];
    Sphere_3 buffer_sphere;
    Point_3 buffer_point;

    std::cout << "==> Number of vertices: " << tess.number_of_vertices() << std::endl;
    std::cout << "==> Number of all cells: " << tess.number_of_cells() << std::endl;
    std::cout << "==> Number of finite cells: " << tess.number_of_finite_cells() << std::endl;
    output->n_simplices = tess.number_of_finite_cells();
    
    output->x.reserve(output->n_simplices);
    output->y.reserve(output->n_simplices);
    output->z.reserve(output->n_simplices);
    output->r.reserve(output->n_simplices);
  
    size_t k = 0;
    
    for (finite_cells cell = tess.finite_cells_begin(); cell != tess.finite_cells_end(); cell++) {
        for (size_t i = 0; i < 4; i++) {
            simplex_vertices[i] = cell->vertex(i)->point();
        }
        
        buffer_sphere = Sphere_3(simplex_vertices[0], simplex_vertices[1], simplex_vertices[2], simplex_vertices[3]);
        buffer_point = cell->circumcenter();
        output->x[k] = CGAL::to_double(buffer_point.x());
        output->y[k] = CGAL::to_double(buffer_point.y());
        output->z[k] = CGAL::to_double(buffer_point.z());
        output->r[k] = CGAL::sqrt(CGAL::to_double(CGAL::squared_distance(buffer_point, cell->vertex(0)->point())));
        k++;
    }
    
    return output;
}

// Full Delaunay with vertex info for DTFE
extern "C" DelaunayOutput* cdelaunay_full_basic(double* X, double* Y, double* Z, size_t n_points) {
    DelaunayOutput* output = new DelaunayOutput();
    std::vector<std::pair<Point, size_t>> points;
    points.reserve(n_points);

    for (size_t i = 0; i < n_points; i++) {
        points.push_back(std::make_pair(Point_3(X[i], Y[i], Z[i]), i));
    }
    
    std::cout << "==> Number of points: " << points.size() << std::endl;
    std::cout << "==> Building Delaunay Triangulation." << std::endl;
    
    DelaunayInfo tess(points.begin(), points.end());
    assert(tess.is_valid());
    points.clear();
    
    Point_3 simplex_vertices[4];
    Sphere_3 buffer_sphere;
    Point_3 buffer_point;
    Tetrahedron_3 buffer_tetrahedron;

    std::cout << "==> Number of vertices: " << tess.number_of_vertices() << std::endl;
    std::cout << "==> Number of all cells: " << tess.number_of_cells() << std::endl;
    std::cout << "==> Number of finite cells: " << tess.number_of_finite_cells() << std::endl;
    output->n_simplices = tess.number_of_finite_cells();
    
    output->x.reserve(output->n_simplices);
    output->y.reserve(output->n_simplices);
    output->z.reserve(output->n_simplices);
    output->r.reserve(output->n_simplices);
    output->volume.reserve(output->n_simplices);
    output->area.reserve(output->n_simplices);
    output->dtfe.resize(tess.number_of_vertices(), 0.0);
  
    size_t k = 0;
    
    for (finite_cells_info cell = tess.finite_cells_begin(); cell != tess.finite_cells_end(); cell++) {
        buffer_tetrahedron = Tetrahedron_3(
            cell->vertex(0)->point(),
            cell->vertex(1)->point(),
            cell->vertex(2)->point(),
            cell->vertex(3)->point()
        );
        
        output->volume[k] = CGAL::to_double(buffer_tetrahedron.volume());
        buffer_point = CGAL::circumcenter(buffer_tetrahedron);
        output->x[k] = CGAL::to_double(buffer_point.x());
        output->y[k] = CGAL::to_double(buffer_point.y());
        output->z[k] = CGAL::to_double(buffer_point.z());
        output->r[k] = CGAL::sqrt(CGAL::to_double(CGAL::squared_distance(buffer_point, cell->vertex(0)->point())));
        
        for (size_t i = 0; i < 4; i++) {
            output->dtfe[cell->vertex(i)->info()] += output->volume[k];
            output->vertices[i].push_back(cell->vertex(i)->info());
        }
        output->area[k] = tetrahedron_area(buffer_tetrahedron);
        
        k++;
    }
    
    return output;
}

// Cleanup function
extern "C" void free_delaunay_output(DelaunayOutput* output) {
    if (output) {
        delete output;
    }
}
