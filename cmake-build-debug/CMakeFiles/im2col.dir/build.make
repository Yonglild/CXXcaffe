# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_COMMAND = /home/wyl/clion-2018.2.3/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/wyl/clion-2018.2.3/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wyl/CXXcaffe

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wyl/CXXcaffe/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/im2col.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/im2col.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/im2col.dir/flags.make

CMakeFiles/im2col.dir/util/im2col.cpp.o: CMakeFiles/im2col.dir/flags.make
CMakeFiles/im2col.dir/util/im2col.cpp.o: ../util/im2col.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wyl/CXXcaffe/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/im2col.dir/util/im2col.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/im2col.dir/util/im2col.cpp.o -c /home/wyl/CXXcaffe/util/im2col.cpp

CMakeFiles/im2col.dir/util/im2col.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/im2col.dir/util/im2col.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wyl/CXXcaffe/util/im2col.cpp > CMakeFiles/im2col.dir/util/im2col.cpp.i

CMakeFiles/im2col.dir/util/im2col.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/im2col.dir/util/im2col.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wyl/CXXcaffe/util/im2col.cpp -o CMakeFiles/im2col.dir/util/im2col.cpp.s

# Object files for target im2col
im2col_OBJECTS = \
"CMakeFiles/im2col.dir/util/im2col.cpp.o"

# External object files for target im2col
im2col_EXTERNAL_OBJECTS =

im2col: CMakeFiles/im2col.dir/util/im2col.cpp.o
im2col: CMakeFiles/im2col.dir/build.make
im2col: CMakeFiles/im2col.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wyl/CXXcaffe/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable im2col"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/im2col.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/im2col.dir/build: im2col

.PHONY : CMakeFiles/im2col.dir/build

CMakeFiles/im2col.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/im2col.dir/cmake_clean.cmake
.PHONY : CMakeFiles/im2col.dir/clean

CMakeFiles/im2col.dir/depend:
	cd /home/wyl/CXXcaffe/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wyl/CXXcaffe /home/wyl/CXXcaffe /home/wyl/CXXcaffe/cmake-build-debug /home/wyl/CXXcaffe/cmake-build-debug /home/wyl/CXXcaffe/cmake-build-debug/CMakeFiles/im2col.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/im2col.dir/depend

