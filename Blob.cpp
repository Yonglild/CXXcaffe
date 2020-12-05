//
// Created by wyl on 2020/11/26.
//

#include "Blob.hpp"

namespace caffe{
    template <typename Dtype>
    void Blob<Dtype>::Reshape(const int num, const int channels, const int height, const int width) {
        vector<int> shape(4);
        shape[0] = num;
        shape[1] = channels;
        shape[2] = height;
        shape[3] = width;
        Reshape(shape);
    }

    template<typename Dtype>
    void Blob<Dtype>::Reshape(const vector<int> &shape) {
        count_ = 1;
        shape_.resize(shape.size());
        if(!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)){
            shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
        }
        int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
        for(int i=0; i < shape.size(); i++){
            count_ *= shape[i];
            shape_[i] = shape[i];
            shape_data[i] = shape[i];
        }
        if(count_ > capacity_){
            capacity_ = count_;
            data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
            diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
        }
    }


}