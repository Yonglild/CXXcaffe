# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

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
CMAKE_COMMAND = /home/wyl/clion-2019.2.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/wyl/clion-2019.2.2/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wyl/CLionProjects/CXXcaffe

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wyl/CLionProjects/CXXcaffe/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/pb_read.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pb_read.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pb_read.dir/flags.make

CMakeFiles/pb_read.dir/protobuf_read.cpp.o: CMakeFiles/pb_read.dir/flags.make
CMakeFiles/pb_read.dir/protobuf_read.cpp.o: ../protobuf_read.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wyl/CLionProjects/CXXcaffe/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pb_read.dir/protobuf_read.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pb_read.dir/protobuf_read.cpp.o -c /home/wyl/CLionProjects/CXXcaffe/protobuf_read.cpp

CMakeFiles/pb_read.dir/protobuf_read.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pb_read.dir/protobuf_read.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wyl/CLionProjects/CXXcaffe/protobuf_read.cpp > CMakeFiles/pb_read.dir/protobuf_read.cpp.i

CMakeFiles/pb_read.dir/protobuf_read.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pb_read.dir/protobuf_read.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wyl/CLionProjects/CXXcaffe/protobuf_read.cpp -o CMakeFiles/pb_read.dir/protobuf_read.cpp.s

# Object files for target pb_read
pb_read_OBJECTS = \
"CMakeFiles/pb_read.dir/protobuf_read.cpp.o"

# External object files for target pb_read
pb_read_EXTERNAL_OBJECTS =

pb_read: CMakeFiles/pb_read.dir/protobuf_read.cpp.o
pb_read: CMakeFiles/pb_read.dir/build.make
pb_read: CMakeFiles/pb_read.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wyl/CLionProjects/CXXcaffe/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pb_read"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pb_read.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pb_read.dir/build: pb_read

.PHONY : CMakeFiles/pb_read.dir/build

CMakeFiles/pb_read.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pb_read.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pb_read.dir/clean

CMakeFiles/pb_read.dir/depend:
	cd /home/wyl/CLionProjects/CXXcaffe/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wyl/CLionProjects/CXXcaffe /home/wyl/CLionProjects/CXXcaffe /home/wyl/CLionProjects/CXXcaffe/cmake-build-debug /home/wyl/CLionProjects/CXXcaffe/cmake-build-debug /home/wyl/CLionProjects/CXXcaffe/cmake-build-debug/CMakeFiles/pb_read.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pb_read.dir/depend
