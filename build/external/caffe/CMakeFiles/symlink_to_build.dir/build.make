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

# Utility rule file for symlink_to_build.

# Include the progress variables for this target.
include external/caffe/CMakeFiles/symlink_to_build.dir/progress.make

external/caffe/CMakeFiles/symlink_to_build:
	$(CMAKE_COMMAND) -E cmake_progress_report /home/fzff9p/c++/Caffe_HelloWorld/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Adding symlink: <caffe_root>/build -> /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe"
	cd /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe && ln -sf /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe /home/fzff9p/c++/Caffe_HelloWorld/external/caffe/build

symlink_to_build: external/caffe/CMakeFiles/symlink_to_build
symlink_to_build: external/caffe/CMakeFiles/symlink_to_build.dir/build.make
.PHONY : symlink_to_build

# Rule to build all files generated by this target.
external/caffe/CMakeFiles/symlink_to_build.dir/build: symlink_to_build
.PHONY : external/caffe/CMakeFiles/symlink_to_build.dir/build

external/caffe/CMakeFiles/symlink_to_build.dir/clean:
	cd /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe && $(CMAKE_COMMAND) -P CMakeFiles/symlink_to_build.dir/cmake_clean.cmake
.PHONY : external/caffe/CMakeFiles/symlink_to_build.dir/clean

external/caffe/CMakeFiles/symlink_to_build.dir/depend:
	cd /home/fzff9p/c++/Caffe_HelloWorld/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fzff9p/c++/Caffe_HelloWorld /home/fzff9p/c++/Caffe_HelloWorld/external/caffe /home/fzff9p/c++/Caffe_HelloWorld/build /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe /home/fzff9p/c++/Caffe_HelloWorld/build/external/caffe/CMakeFiles/symlink_to_build.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : external/caffe/CMakeFiles/symlink_to_build.dir/depend

