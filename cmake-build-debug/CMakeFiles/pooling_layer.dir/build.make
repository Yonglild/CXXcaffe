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
include CMakeFiles/pooling_layer.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pooling_layer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pooling_layer.dir/flags.make

CMakeFiles/pooling_layer.dir/layers/pooling_layer.cpp.o: CMakeFiles/pooling_layer.dir/flags.make
CMakeFiles/pooling_layer.dir/layers/pooling_layer.cpp.o: ../layers/pooling_layer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wyl/CXXcaffe/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pooling_layer.dir/layers/pooling_layer.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pooling_layer.dir/layers/pooling_layer.cpp.o -c /home/wyl/CXXcaffe/layers/pooling_layer.cpp

CMakeFiles/pooling_layer.dir/layers/pooling_layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pooling_layer.dir/layers/pooling_layer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wyl/CXXcaffe/layers/pooling_layer.cpp > CMakeFiles/pooling_layer.dir/layers/pooling_layer.cpp.i

CMakeFiles/pooling_layer.dir/layers/pooling_layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pooling_layer.dir/layers/pooling_layer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wyl/CXXcaffe/layers/pooling_layer.cpp -o CMakeFiles/pooling_layer.dir/layers/pooling_layer.cpp.s

# Object files for target pooling_layer
pooling_layer_OBJECTS = \
"CMakeFiles/pooling_layer.dir/layers/pooling_layer.cpp.o"

# External object files for target pooling_layer
pooling_layer_EXTERNAL_OBJECTS =

pooling_layer: CMakeFiles/pooling_layer.dir/layers/pooling_layer.cpp.o
pooling_layer: CMakeFiles/pooling_layer.dir/build.make
pooling_layer: CMakeFiles/pooling_layer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wyl/CXXcaffe/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pooling_layer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pooling_layer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pooling_layer.dir/build: pooling_layer

.PHONY : CMakeFiles/pooling_layer.dir/build

CMakeFiles/pooling_layer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pooling_layer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pooling_layer.dir/clean

CMakeFiles/pooling_layer.dir/depend:
	cd /home/wyl/CXXcaffe/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wyl/CXXcaffe /home/wyl/CXXcaffe /home/wyl/CXXcaffe/cmake-build-debug /home/wyl/CXXcaffe/cmake-build-debug /home/wyl/CXXcaffe/cmake-build-debug/CMakeFiles/pooling_layer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pooling_layer.dir/depend
