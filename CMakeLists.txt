cmake_minimum_required(VERSION 3.10)
project(cxxbasic)

include_directories(net.hpp Blob.hpp)
add_executable(net net.cpp)
add_executable(Blob Blob.cpp)
