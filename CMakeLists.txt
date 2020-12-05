cmake_minimum_required(VERSION 3.10)
project(cxxbasic)

include_directories(net.hpp Blob.hpp SyncedMemory.hpp)
add_executable(net net.cpp)
add_executable(Blob Blob.cpp)
add_executable(SyncedMemory SyncedMemory.cpp)



add_executable(shared_ptr shared_ptr.cpp)
