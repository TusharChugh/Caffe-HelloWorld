# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/fzff9p/c++/Caffe_HelloWorld

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/fzff9p/c++/Caffe_HelloWorld/build

# Utility rule file for uninstall.

# Include the progress variables for this target.
include external/caffe/CMakeFiles/uninstall.dir/progress.make

external/caffe/CMakeFiles/uninstall:
	cd /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe && /usr/bin/cmake -P /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/cmake/Uninstall.cmake

uninstall: external/caffe/CMakeFiles/uninstall
uninstall: external/caffe/CMakeFiles/uninstall.dir/build.make
.PHONY : uninstall

# Rule to build all files generated by this target.
external/caffe/CMakeFiles/uninstall.dir/build: uninstall
.PHONY : external/caffe/CMakeFiles/uninstall.dir/build

external/caffe/CMakeFiles/uninstall.dir/clean:
	cd /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe && $(CMAKE_COMMAND) -P CMakeFiles/uninstall.dir/cmake_clean.cmake
.PHONY : external/caffe/CMakeFiles/uninstall.dir/clean

external/caffe/CMakeFiles/uninstall.dir/depend:
	cd /home/fzff9p/c++/Caffe_HelloWorld/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fzff9p/c++/Caffe_HelloWorld /home/fzff9p/c++/Caffe_HelloWorld/external/caffe /home/fzff9p/c++/Caffe_HelloWorld/build /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/CMakeFiles/uninstall.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : external/caffe/CMakeFiles/uninstall.dir/depend
