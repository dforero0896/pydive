# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/ebsofts/CMake/3.22.1-GCCcore-11.2.0/bin/cmake

# The command to remove a file.
RM = /opt/ebsofts/CMake/3.22.1-GCCcore-11.2.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/astro/dforero/codes/pydive

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/astro/dforero/codes/pydive

# Utility rule file for ReplicatePythonSourceTree.

# Include any custom commands dependencies for this target.
include CMakeFiles/ReplicatePythonSourceTree.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ReplicatePythonSourceTree.dir/progress.make

CMakeFiles/ReplicatePythonSourceTree:
	/opt/ebsofts/CMake/3.22.1-GCCcore-11.2.0/bin/cmake -P /home/astro/dforero/codes/pydive/cmake/ReplicatePythonSourceTree.cmake /home/astro/dforero/codes/pydive

ReplicatePythonSourceTree: CMakeFiles/ReplicatePythonSourceTree
ReplicatePythonSourceTree: CMakeFiles/ReplicatePythonSourceTree.dir/build.make
.PHONY : ReplicatePythonSourceTree

# Rule to build all files generated by this target.
CMakeFiles/ReplicatePythonSourceTree.dir/build: ReplicatePythonSourceTree
.PHONY : CMakeFiles/ReplicatePythonSourceTree.dir/build

CMakeFiles/ReplicatePythonSourceTree.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ReplicatePythonSourceTree.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ReplicatePythonSourceTree.dir/clean

CMakeFiles/ReplicatePythonSourceTree.dir/depend:
	cd /home/astro/dforero/codes/pydive && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/astro/dforero/codes/pydive /home/astro/dforero/codes/pydive /home/astro/dforero/codes/pydive /home/astro/dforero/codes/pydive /home/astro/dforero/codes/pydive/CMakeFiles/ReplicatePythonSourceTree.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ReplicatePythonSourceTree.dir/depend

