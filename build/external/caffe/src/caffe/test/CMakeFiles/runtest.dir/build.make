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

# Utility rule file for runtest.

# Include the progress variables for this target.
include external/caffe/src/caffe/test/CMakeFiles/runtest.dir/progress.make

external/caffe/src/caffe/test/CMakeFiles/runtest:
	cd /home/fzff9p/c++/Caffe_HelloWorld/external/caffe && /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/test/test.testbin --gtest_shuffle

runtest: external/caffe/src/caffe/test/CMakeFiles/runtest
runtest: external/caffe/src/caffe/test/CMakeFiles/runtest.dir/build.make
.PHONY : runtest

# Rule to build all files generated by this target.
external/caffe/src/caffe/test/CMakeFiles/runtest.dir/build: runtest
.PHONY : external/caffe/src/caffe/test/CMakeFiles/runtest.dir/build

external/caffe/src/caffe/test/CMakeFiles/runtest.dir/clean:
	cd /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/src/caffe/test && $(CMAKE_COMMAND) -P CMakeFiles/runtest.dir/cmake_clean.cmake
.PHONY : external/caffe/src/caffe/test/CMakeFiles/runtest.dir/clean

external/caffe/src/caffe/test/CMakeFiles/runtest.dir/depend:
	cd /home/fzff9p/c++/Caffe_HelloWorld/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fzff9p/c++/Caffe_HelloWorld /home/fzff9p/c++/Caffe_HelloWorld/external/caffe/src/caffe/test /home/fzff9p/c++/Caffe_HelloWorld/build /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/src/caffe/test /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/src/caffe/test/CMakeFiles/runtest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : external/caffe/src/caffe/test/CMakeFiles/runtest.dir/depend
