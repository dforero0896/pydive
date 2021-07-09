#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_3.h>
#include <CGAL/Sphere_3.h>
#include <CGAL/Tetrahedron_3.h>

#include <vector>
#include <set>
#include <cassert>
#include <algorithm>

typedef CGAL::Exact_predicates_exact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_3<size_t,K> Vb;
typedef CGAL::Triangulation_data_structure_3<Vb>  Tds;
typedef CGAL::Delaunay_triangulation_3<K,Tds> Delaunay;
typedef CGAL::Periodic_3_Delaunay_triangulation_3<Vb> PDelaunay;
typedef Delaunay::Point Point;
typedef Delaunay::Locate_type Locate_type;
typedef Delaunay::Cell_handle Cell_handle;
typedef Delaunay::Cell Cell;
typedef Delaunay::Vertex_handle Vertex_handle;

typedef K::FT FT;
typedef CGAL::Point_3<K> Point_3;
typedef CGAL::Sphere_3<K> Sphere_3;
typedef CGAL::Tetrahedron_3<K> Tetrahedron_3;

struct VertexList{
    std::vector<size_t> v;
};

class DelaunayOutput{
    public:
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> z;
        std::vector<double> r;
        std::vector<double> volume;
        std::vector<double> dtfe;
        std::vector<double> weights;
        std::vector<double> selection;
        std::vector<size_t> vertices[4];
        size_t n_simplices;
        

};


void tesselation_to_voids(DelaunayOutput *output, Delaunay* tess){
    Point_3 simplex_vertices[4];
    Sphere_3 buffer_sphere;
    Point_3 buffer_point;

    std::cout<<"==> Number of vertices: "<<tess->number_of_vertices()<<std::endl;
    std::cout<<"==> Number of all cells: "<<tess->number_of_cells()<<std::endl;
    std::cout<<"==> Number of finite cells: "<<tess->number_of_finite_cells()<<std::endl;
    output->n_simplices = tess->number_of_finite_cells();
    
    output->x.reserve(tess->number_of_finite_cells());
    output->y.reserve(tess->number_of_finite_cells());
    output->z.reserve(tess->number_of_finite_cells());
    output->r.reserve(tess->number_of_finite_cells());
  
    std::size_t k = 0;
    std::size_t i;
    for(Delaunay::Finite_cells_iterator cell=tess->finite_cells_begin();cell!=tess->finite_cells_end();cell++) {
        
        for(i=0;i<4;i++){
            simplex_vertices[i] = cell->vertex(i)->point();
            output->vertices[i].push_back(cell->vertex(i)->info());
        }
        
        buffer_sphere = Sphere_3(simplex_vertices[0],simplex_vertices[1],simplex_vertices[2],simplex_vertices[3]);
        buffer_point = buffer_sphere.center();
        output->x[k] = CGAL::to_double(buffer_point.x());
        output->y[k] = CGAL::to_double(buffer_point.y());
        output->z[k] = CGAL::to_double(buffer_point.z());
        output->r[k] = CGAL::sqrt(CGAL::to_double(buffer_sphere.squared_radius()));

        k++;
    }
}

void tesselation_to_voids_w_dtfe(DelaunayOutput *output, Delaunay* tess){
    Point_3 simplex_vertices[4];
    Sphere_3 buffer_sphere;
    Point_3 buffer_point;
    Tetrahedron_3 buffer_tetrahedron;

    std::cout<<"==> Number of vertices: "<<tess->number_of_vertices()<<std::endl;
    std::cout<<"==> Number of all cells: "<<tess->number_of_cells()<<std::endl;
    std::cout<<"==> Number of finite cells: "<<tess->number_of_finite_cells()<<std::endl;
    output->n_simplices = tess->number_of_finite_cells();
    
    output->x.reserve(tess->number_of_finite_cells());
    output->y.reserve(tess->number_of_finite_cells());
    output->z.reserve(tess->number_of_finite_cells());
    output->r.reserve(tess->number_of_finite_cells());
    output->volume.reserve(tess->number_of_finite_cells());
    output->dtfe.reserve(tess->number_of_vertices());
  
    std::size_t k = 0;
    std::size_t i;
    for(i=0; i<tess->number_of_vertices();i++){
        output->dtfe[i] = 0;
    }
    for(Delaunay::Finite_cells_iterator cell=tess->finite_cells_begin();cell!=tess->finite_cells_end();cell++) {
        
        for(i=0;i<4;i++){
            simplex_vertices[i] = cell->vertex(i)->point();
            output->vertices[i].push_back(cell->vertex(i)->info());
        }
        
        buffer_sphere = Sphere_3(simplex_vertices[0],simplex_vertices[1],simplex_vertices[2],simplex_vertices[3]);
        buffer_point = buffer_sphere.center();
        output->x[k] = CGAL::to_double(buffer_point.x());
        output->y[k] = CGAL::to_double(buffer_point.y());
        output->z[k] = CGAL::to_double(buffer_point.z());
        output->r[k] = CGAL::sqrt(CGAL::to_double(buffer_sphere.squared_radius()));

        buffer_tetrahedron = Tetrahedron_3(simplex_vertices[0],simplex_vertices[1],simplex_vertices[2],simplex_vertices[3]);
        output->volume[k] = CGAL::to_double(buffer_tetrahedron.volume());

        for(i=0;i<4;i++){
            output->dtfe[cell->vertex(i)->info()] += output->volume[k];
        }


        k++;
    }
}

DelaunayOutput cdelaunay(std::vector<double> X, std::vector<double> Y, std::vector<double> Z, bool compute_dtfe){

    DelaunayOutput output;
    std::vector< std::pair<Point,size_t> >points;
    
    
    

    std::size_t i, n_points;
    n_points = X.size();
    for (i=0; i<n_points; i++){
        points.push_back(std::make_pair(Point_3(X[i], Y[i], Z[i]),i));
    }
    
    std::cout<<"==> Number of points: "<<points.size()<<std::endl;
    std::cout<<"==> Building Delaunay Triangulation."<<std::endl;
    Delaunay tess(points.begin(), points.end());
    assert(tess.is_valid());
    points.clear();
    
    
    if (compute_dtfe){
        tesselation_to_voids_w_dtfe(&output, &tess);
    }
    else{
        tesselation_to_voids(&output, &tess);
    }
    
    return output;

}


DelaunayOutput cdelaunay_periodic(std::vector<double> X, std::vector<double> Y, std::vector<double> Z, double box_size, double cpy_range, bool compute_dtfe){

    double low_range = 0;
    double high_range = low_range + box_size;
    std::size_t size;
    DelaunayOutput output;
    std::vector< std::pair<Point,size_t> >points;
    
    std::size_t i, n_points;
    n_points = X.size();
    for (i=0; i<n_points; i++){
        points.push_back(std::make_pair(Point_3(X[i], Y[i], Z[i]),i));
    }
    size = points.size();
    std::cout<<"==> Number of points: "<<points.size()<<std::endl<<std::endl;
    size_t point_count = n_points;
    std::cout<<"Duplicating boundaries for periodic condition"<<std::endl;
    for(i=0;i<size;i++)
        if(points[i].first.x()<low_range+cpy_range){
            points.push_back(std::make_pair(Point_3(points[i].first.x()+box_size,points[i].first.y(),points[i].first.z()),point_count++));
        }
    size=points.size();
    for(i=0;i<size;i++) 
        if(points[i].first.x()>=high_range-cpy_range && points[i].first.x()<high_range){
            points.push_back(std::make_pair(Point_3(points[i].first.x()-box_size,points[i].first.y(),points[i].first.z()),point_count++));
        }
    size=points.size();
    for(i=0;i<size;i++) 
        if(points[i].first.y()<low_range+cpy_range){
            points.push_back(std::make_pair(Point_3(points[i].first.x(),points[i].first.y()+box_size,points[i].first.z()),point_count++));
        }
    size=points.size();
    for(i=0;i<size;i++) 
        if(points[i].first.y()>=high_range-cpy_range && points[i].first.y()<high_range){
            points.push_back(std::make_pair(Point_3(points[i].first.x(),points[i].first.y()-box_size,points[i].first.z()),point_count++));
        }
    size=points.size();
    for(i=0;i<size;i++) 
        if(points[i].first.z()<low_range+cpy_range){
            points.push_back(std::make_pair(Point_3(points[i].first.x(),points[i].first.y(),points[i].first.z()+box_size),point_count++));
        }
    size=points.size();
    for(i=0;i<size;i++) 
        if(points[i].first.z()>=high_range-cpy_range && points[i].first.z()<high_range){
            points.push_back(std::make_pair(Point_3(points[i].first.x(),points[i].first.y(),points[i].first.z()-box_size),point_count++));
        }
    size=points.size();

    std::cout<<"==> Number of points: "<<points.size()<<std::endl<<std::endl;

    std::cout<<"==> Building Delaunay Triangulation."<<std::endl;
    Delaunay tess(points.begin(), points.end());
    assert(tess.is_valid());
    points.clear();
    
    if (compute_dtfe){
        tesselation_to_voids_w_dtfe(&output, &tess);
    }
    else{
        tesselation_to_voids(&output, &tess);
    }
    return output;

}

int main(){
    return 0;
}

