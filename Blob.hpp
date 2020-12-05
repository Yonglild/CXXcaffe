//
// Created by wyl on 2020/11/26.
//
#ifndef CXXBASIC_BLOB_H
#define CXXBASIC_BLOB_H

#include <vector>
#include <string>
#include <sstream>
#include <memory>
#include "SyncedMemory.hpp"
using namespace std;

const int kMaxBlobAxes = 32;

namespace caffe{
    template <typename Dtype>
    class Blob{
    public:
        explicit Blob(const int num, const int channels, const int height ,const int width);
        explicit Blob(const vector<int>& shape);
        void Reshape(const int num, const int channels, const int height, const int width);
        void Reshape(const vector<int>& shape);

        inline string shape_string() const {
            std::stringstream oss;
            for(int i=0; i<shape_.size(); i++){
                oss << shape_[i] << " ";
            }
            oss << "(" << count_ << ")";
            return oss.str();
        }

        inline int num_axes() const {
            return shape_.size();
        }

        inline int count() const{
            return count_;
        }

    protected:
        shared_ptr<SyncedMemory> shape_data_;       // SyncedMemory为CPU和GPU共用内存
        shared_ptr<SyncedMemory> data_;       // SyncedMemory为CPU和GPU共用内存
        shared_ptr<SyncedMemory> diff_;       // SyncedMemory为CPU和GPU共用内存
        vector<int> shape_;
        int count_;
        int capacity_;
    };
}

#endif //CXXBASIC_BLOB_H
