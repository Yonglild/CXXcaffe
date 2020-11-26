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
        if(){

        }
    }


}