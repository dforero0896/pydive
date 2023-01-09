# Created by the script cgal_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.

cmake_minimum_required(VERSION 3.1...3.23)

project( pydive )
list(APPEND CMAKE_MODULE_PATH "${CGAL_DIR}/cmake/modules/")
find_package( TBB REQUIRED)


# CGAL and its components
find_package( CGAL QUIET COMPONENTS  )
include(${CGAL_USE_FILE})

if ( NOT CGAL_FOUND )

  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()  

endif()


# Boost and its components
find_package( Boost REQUIRED )

if ( NOT Boost_FOUND )

  message(STATUS "This project requires the Boost library, and will not be compiled.")

  return()  

endif()

# include for local directory

# include for local package
include(CGAL_TBB_support)



# Creating entries for all C++ files with "main" routine
# ##########################################################


#create_single_source_cgal_program( "delaunay_backend.cpp" )

# Creating entries for target: delaunay_backend
# ############################
#add_executable( delaunay_backend  delaunay_backend.cpp )

# If the pyx file is a C++ file, we should specify that here.
set_source_files_properties(
  ${CMAKE_CURRENT_LIST_DIR}/pydive.pyx
  PROPERTIES CYTHON_IS_CXX TRUE )

# Multi-file cython modules do not appear to be working at the moment.
cython_add_module( pydive pydive.pyx  )

#add_to_cached_list( CGAL_EXECUTABLE_TARGETS delaunay_backend )

# Link the executable to CGAL and third-party libraries
#target_link_libraries(delaunay_backend PRIVATE CGAL::CGAL CGAL::TBB_support)
#target_link_libraries(pydive PRIVATE CGAL::CGAL CGAL::TBB_support)



