# Defines the CMake commands/policies
cmake_minimum_required(VERSION 3.1...3.23)

# Set the project name
project( pydive )

# Make the scripts available in the 'cmake' directory available for the
# 'include()' command, 'find_package()' command.
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )
list(APPEND CMAKE_MODULE_PATH "${CGAL_DIR}/cmake/modules/")
add_definitions( -DCGAL_TRIANGULATION_3_PROFILING)
find_package( TBB REQUIRED)
find_package( OpenMP REQUIRED)


# Include the CMake script UseCython.cmake.  This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
include( UseCython )

# With CMake, a clean separation can be made between the source tree and the
# build tree.  When all source is compiled, as with pure C/C++, the source is
# no-longer needed in the build tree.  However, with pure *.py source, the
# source is processed directly.  To handle this, we reproduce the availability
# of the source files in the build tree.
add_custom_target( ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReplicatePythonSourceTree.cmake
  ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

include_directories( ${CMAKE_CURRENT_LIST_DIR}/pydive ${TBB_INCLUDE_DIRS})
set_source_files_properties(
  ${CMAKE_CURRENT_LIST_DIR}/pydive/pydive.pyx
  PROPERTIES CYTHON_IS_CXX TRUE)

# Process the CMakeLists.txt in the 'src' and 'bin' directory.
add_subdirectory( pydive )

