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
include external/caffe/src/caffe/CMakeFiles/proto.dir/depend.make

# Include the progress variables for this target.
include external/caffe/src/caffe/CMakeFiles/proto.dir/progress.make

# Include the compile flags for this target's objects.
include external/caffe/src/caffe/CMakeFiles/proto.dir/flags.make

external/caffe/include/caffe/proto/caffe.pb.cc: ../external/caffe/src/caffe/proto/caffe.proto
	$(CMAKE_COMMAND) -E cmake_progress_report /home/fzff9p/c++/Caffe_HelloWorld/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Running C++/Python protocol buffer compiler on /home/fzff9p/c++/Caffe_HelloWorld/external/caffe/src/caffe/proto/caffe.proto"
	cd /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/src/caffe && /usr/bin/cmake -E make_directory /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/include/caffe/proto
	cd /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/src/caffe && /usr/bin/protoc --cpp_out /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/include/caffe/proto -I /home/fzff9p/c++/Caffe_HelloWorld/external/caffe/src/caffe/proto /home/fzff9p/c++/Caffe_HelloWorld/external/caffe/src/caffe/proto/caffe.proto
	cd /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/src/caffe && /usr/bin/protoc --python_out /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/include/caffe/proto -I /home/fzff9p/c++/Caffe_HelloWorld/external/caffe/src/caffe/proto /home/fzff9p/c++/Caffe_HelloWorld/external/caffe/src/caffe/proto/caffe.proto

external/caffe/include/caffe/proto/caffe.pb.h: external/caffe/include/caffe/proto/caffe.pb.cc

external/caffe/include/caffe/proto/caffe_pb2.py: external/caffe/include/caffe/proto/caffe.pb.cc

external/caffe/src/caffe/CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.o: external/caffe/src/caffe/CMakeFiles/proto.dir/flags.make
external/caffe/src/caffe/CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.o: external/caffe/include/caffe/proto/caffe.pb.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /home/fzff9p/c++/Caffe_HelloWorld/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object external/caffe/src/caffe/CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.o"
	cd /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/src/caffe && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.o -c /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/include/caffe/proto/caffe.pb.cc

external/caffe/src/caffe/CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.i"
	cd /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/src/caffe && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/include/caffe/proto/caffe.pb.cc > CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.i

external/caffe/src/caffe/CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.s"
	cd /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/src/caffe && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/include/caffe/proto/caffe.pb.cc -o CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.s

external/caffe/src/caffe/CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.requires:
.PHONY : external/caffe/src/caffe/CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.requires

external/caffe/src/caffe/CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.provides: external/caffe/src/caffe/CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.requires
	$(MAKE) -f external/caffe/src/caffe/CMakeFiles/proto.dir/build.make external/caffe/src/caffe/CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.provides.build
.PHONY : external/caffe/src/caffe/CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.provides

external/caffe/src/caffe/CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.provides.build: external/caffe/src/caffe/CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.o

# Object files for target proto
proto_OBJECTS = \
"CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.o"

# External object files for target proto
proto_EXTERNAL_OBJECTS =

external/caffe/lib/libproto.a: external/caffe/src/caffe/CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.o
external/caffe/lib/libproto.a: external/caffe/src/caffe/CMakeFiles/proto.dir/build.make
external/caffe/lib/libproto.a: external/caffe/src/caffe/CMakeFiles/proto.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library ../../lib/libproto.a"
	cd /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/src/caffe && $(CMAKE_COMMAND) -P CMakeFiles/proto.dir/cmake_clean_target.cmake
	cd /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/src/caffe && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/proto.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/caffe/src/caffe/CMakeFiles/proto.dir/build: external/caffe/lib/libproto.a
.PHONY : external/caffe/src/caffe/CMakeFiles/proto.dir/build

external/caffe/src/caffe/CMakeFiles/proto.dir/requires: external/caffe/src/caffe/CMakeFiles/proto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.requires
.PHONY : external/caffe/src/caffe/CMakeFiles/proto.dir/requires

external/caffe/src/caffe/CMakeFiles/proto.dir/clean:
	cd /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/src/caffe && $(CMAKE_COMMAND) -P CMakeFiles/proto.dir/cmake_clean.cmake
.PHONY : external/caffe/src/caffe/CMakeFiles/proto.dir/clean

external/caffe/src/caffe/CMakeFiles/proto.dir/depend: external/caffe/include/caffe/proto/caffe.pb.cc
external/caffe/src/caffe/CMakeFiles/proto.dir/depend: external/caffe/include/caffe/proto/caffe.pb.h
external/caffe/src/caffe/CMakeFiles/proto.dir/depend: external/caffe/include/caffe/proto/caffe_pb2.py
	cd /home/fzff9p/c++/Caffe_HelloWorld/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fzff9p/c++/Caffe_HelloWorld /home/fzff9p/c++/Caffe_HelloWorld/external/caffe/src/caffe /home/fzff9p/c++/Caffe_HelloWorld/build /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/src/caffe /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/src/caffe/CMakeFiles/proto.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : external/caffe/src/caffe/CMakeFiles/proto.dir/depend

