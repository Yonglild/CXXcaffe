cmake_minimum_required(VERSION 3.10)
project(cxxbasic)
set(CMAKE_CXX_STANDARD 11)      # 必须加上，否则shared_ptr不能编译通过
set(${CAFFE_INCLUDE_DIR} ./include)
include_directories(${CAFFE_INCLUDE_DIR}
        ${CAFFE_INCLUDE_DIR}/util/im2col.hpp
        ${CAFFE_INCLUDE_DIR}/util/math_functions.hpp
        ${CAFFE_INCLUDE_DIR}/layers/base_conv_layer.hpp
        ${CAFFE_INCLUDE_DIR}/layers/conv_layer.hpp
        ${CAFFE_INCLUDE_DIR}/layers/pooling_layer.hpp
        ${CAFFE_INCLUDE_DIR}/proto/caffe.pb.h)

add_executable(net net.cpp)
add_executable(Blob Blob.cpp)
add_executable(SyncedMemory SyncedMemory.cpp)
add_executable(layer layer.cpp)
add_executable(shared_ptr shared_ptr.cpp)

add_executable(base_conv_layer layers/base_conv_layer.cpp)
add_executable(conv_layer layers/conv_layer.cpp)
add_executable(pooling_layer layers/pooling_layer.cpp)
add_executable(im2col util/im2col.cpp)
