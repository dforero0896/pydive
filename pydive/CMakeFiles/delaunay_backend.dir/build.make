# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /global/u1/d/dforero/codes/pydive/pydive

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /global/u1/d/dforero/codes/pydive/pydive

# Include any dependencies generated for this target.
include CMakeFiles/delaunay_backend.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/delaunay_backend.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/delaunay_backend.dir/flags.make

CMakeFiles/delaunay_backend.dir/delaunay_backend.cpp.o: CMakeFiles/delaunay_backend.dir/flags.make
CMakeFiles/delaunay_backend.dir/delaunay_backend.cpp.o: delaunay_backend.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/u1/d/dforero/codes/pydive/pydive/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/delaunay_backend.dir/delaunay_backend.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/delaunay_backend.dir/delaunay_backend.cpp.o -c /global/u1/d/dforero/codes/pydive/pydive/delaunay_backend.cpp

CMakeFiles/delaunay_backend.dir/delaunay_backend.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/delaunay_backend.dir/delaunay_backend.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /global/u1/d/dforero/codes/pydive/pydive/delaunay_backend.cpp > CMakeFiles/delaunay_backend.dir/delaunay_backend.cpp.i

CMakeFiles/delaunay_backend.dir/delaunay_backend.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/delaunay_backend.dir/delaunay_backend.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /global/u1/d/dforero/codes/pydive/pydive/delaunay_backend.cpp -o CMakeFiles/delaunay_backend.dir/delaunay_backend.cpp.s

# Object files for target delaunay_backend
delaunay_backend_OBJECTS = \
"CMakeFiles/delaunay_backend.dir/delaunay_backend.cpp.o"

# External object files for target delaunay_backend
delaunay_backend_EXTERNAL_OBJECTS =

delaunay_backend: CMakeFiles/delaunay_backend.dir/delaunay_backend.cpp.o
delaunay_backend: CMakeFiles/delaunay_backend.dir/build.make
delaunay_backend: /global/homes/d/dforero/.conda/envs/nbk-env/lib/libgmpxx.so
delaunay_backend: /global/homes/d/dforero/.conda/envs/nbk-env/lib/libmpfr.so
delaunay_backend: /global/homes/d/dforero/.conda/envs/nbk-env/lib/libgmp.so
delaunay_backend: CMakeFiles/delaunay_backend.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/global/u1/d/dforero/codes/pydive/pydive/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable delaunay_backend"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/delaunay_backend.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/delaunay_backend.dir/build: delaunay_backend

.PHONY : CMakeFiles/delaunay_backend.dir/build

CMakeFiles/delaunay_backend.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/delaunay_backend.dir/cmake_clean.cmake
.PHONY : CMakeFiles/delaunay_backend.dir/clean

CMakeFiles/delaunay_backend.dir/depend:
	cd /global/u1/d/dforero/codes/pydive/pydive && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /global/u1/d/dforero/codes/pydive/pydive /global/u1/d/dforero/codes/pydive/pydive /global/u1/d/dforero/codes/pydive/pydive /global/u1/d/dforero/codes/pydive/pydive /global/u1/d/dforero/codes/pydive/pydive/CMakeFiles/delaunay_backend.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/delaunay_backend.dir/depend

