cmake_minimum_required(VERSION 3.10)
project(cxxbasic)
set(CMAKE_CXX_STANDARD 11)      # 必须加上，否则shared_ptr不能编译通过
set(${CAFFE_INCLUDE_DIR} ./include)
include_directories(${CAFFE_INCLUDE_DIR}
        ${CAFFE_INCLUDE_DIR}/proto)

add_executable(net net.cpp)
add_executable(Blob Blob.cpp)
add_executable(SyncedMemory SyncedMemory.cpp)


add_executable(shared_ptr shared_ptr.cpp)
