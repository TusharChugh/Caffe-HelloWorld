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

# Include any dependencies generated for this target.
include CMakeFiles/Caffe_HelloWorld.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Caffe_HelloWorld.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Caffe_HelloWorld.dir/flags.make

CMakeFiles/Caffe_HelloWorld.dir/main.cpp.o: CMakeFiles/Caffe_HelloWorld.dir/flags.make
CMakeFiles/Caffe_HelloWorld.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/fzff9p/c++/Caffe_HelloWorld/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Caffe_HelloWorld.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Caffe_HelloWorld.dir/main.cpp.o -c /home/fzff9p/c++/Caffe_HelloWorld/main.cpp

CMakeFiles/Caffe_HelloWorld.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Caffe_HelloWorld.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/fzff9p/c++/Caffe_HelloWorld/main.cpp > CMakeFiles/Caffe_HelloWorld.dir/main.cpp.i

CMakeFiles/Caffe_HelloWorld.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Caffe_HelloWorld.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/fzff9p/c++/Caffe_HelloWorld/main.cpp -o CMakeFiles/Caffe_HelloWorld.dir/main.cpp.s

CMakeFiles/Caffe_HelloWorld.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/Caffe_HelloWorld.dir/main.cpp.o.requires

CMakeFiles/Caffe_HelloWorld.dir/main.cpp.o.provides: CMakeFiles/Caffe_HelloWorld.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/Caffe_HelloWorld.dir/build.make CMakeFiles/Caffe_HelloWorld.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/Caffe_HelloWorld.dir/main.cpp.o.provides

CMakeFiles/Caffe_HelloWorld.dir/main.cpp.o.provides.build: CMakeFiles/Caffe_HelloWorld.dir/main.cpp.o

# Object files for target Caffe_HelloWorld
Caffe_HelloWorld_OBJECTS = \
"CMakeFiles/Caffe_HelloWorld.dir/main.cpp.o"

# External object files for target Caffe_HelloWorld
Caffe_HelloWorld_EXTERNAL_OBJECTS =

Caffe_HelloWorld: CMakeFiles/Caffe_HelloWorld.dir/main.cpp.o
Caffe_HelloWorld: CMakeFiles/Caffe_HelloWorld.dir/build.make
Caffe_HelloWorld: external/caffe/lib/libcaffe.so.1.0.0
Caffe_HelloWorld: external/caffe/lib/libproto.a
Caffe_HelloWorld: /usr/lib/x86_64-linux-gnu/libboost_system.so
Caffe_HelloWorld: /usr/lib/x86_64-linux-gnu/libboost_thread.so
Caffe_HelloWorld: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
Caffe_HelloWorld: /usr/lib/x86_64-linux-gnu/libpthread.so
Caffe_HelloWorld: /usr/lib/x86_64-linux-gnu/libglog.so
Caffe_HelloWorld: /usr/lib/x86_64-linux-gnu/libgflags.so
Caffe_HelloWorld: /usr/lib/x86_64-linux-gnu/libprotobuf.so
Caffe_HelloWorld: /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
Caffe_HelloWorld: /usr/lib/x86_64-linux-gnu/libhdf5.so
Caffe_HelloWorld: /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
Caffe_HelloWorld: /usr/lib/x86_64-linux-gnu/libhdf5.so
Caffe_HelloWorld: /usr/lib/x86_64-linux-gnu/liblmdb.so
Caffe_HelloWorld: /usr/lib/x86_64-linux-gnu/libleveldb.so
Caffe_HelloWorld: /usr/local/cuda-8.0/lib64/libcudart.so
Caffe_HelloWorld: /usr/local/cuda-8.0/lib64/libcurand.so
Caffe_HelloWorld: /usr/local/cuda-8.0/lib64/libcublas.so
Caffe_HelloWorld: /usr/local/cuda-8.0/lib64/libcudnn.so
Caffe_HelloWorld: /usr/local/lib/libopencv_highgui.so.2.4.13
Caffe_HelloWorld: /usr/local/lib/libopencv_imgproc.so.2.4.13
Caffe_HelloWorld: /usr/local/lib/libopencv_core.so.2.4.13
Caffe_HelloWorld: /usr/local/cuda-8.0/lib64/libcudart.so
Caffe_HelloWorld: /usr/local/cuda-8.0/lib64/libnppc.so
Caffe_HelloWorld: /usr/local/cuda-8.0/lib64/libnppi.so
Caffe_HelloWorld: /usr/local/cuda-8.0/lib64/libnpps.so
Caffe_HelloWorld: /usr/lib/liblapack.so
Caffe_HelloWorld: /usr/lib/libcblas.so
Caffe_HelloWorld: /usr/lib/libatlas.so
Caffe_HelloWorld: /usr/lib/x86_64-linux-gnu/libboost_python.so
Caffe_HelloWorld: CMakeFiles/Caffe_HelloWorld.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable Caffe_HelloWorld"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Caffe_HelloWorld.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Caffe_HelloWorld.dir/build: Caffe_HelloWorld
.PHONY : CMakeFiles/Caffe_HelloWorld.dir/build

CMakeFiles/Caffe_HelloWorld.dir/requires: CMakeFiles/Caffe_HelloWorld.dir/main.cpp.o.requires
.PHONY : CMakeFiles/Caffe_HelloWorld.dir/requires

CMakeFiles/Caffe_HelloWorld.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Caffe_HelloWorld.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Caffe_HelloWorld.dir/clean

CMakeFiles/Caffe_HelloWorld.dir/depend:
	cd /home/fzff9p/c++/Caffe_HelloWorld/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fzff9p/c++/Caffe_HelloWorld /home/fzff9p/c++/Caffe_HelloWorld /home/fzff9p/c++/Caffe_HelloWorld/build /home/fzff9p/c++/Caffe_HelloWorld/build /home/fzff9p/c++/Caffe_HelloWorld/build/CMakeFiles/Caffe_HelloWorld.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Caffe_HelloWorld.dir/depend
