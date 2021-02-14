//
// Created by wyl on 21-2-14.
//

#ifndef CXXBASIC_POOLING_LAYER_H
#define CXXBASIC_POOLING_LAYER_H

#include <vector>
#include "../Blob.hpp"
#include "../layer.hpp"
#include "../proto/caffe.pb.h"

namespace caffe{
template <typename Dtype>
class PoolingLayer:public Layer<Dtype>{
public:
    explicit PoolingLayer(const LayerParameter& param)
    : Layer(param){}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
    int channels_;
    int height_, width_;
    int pooled_height_, pooled_width_;
    bool global_pooling_;
    PoolingParameter_RoundMode round_mode_;
    Blob<Dtype> rand_idx_;
    Blob<int> max_idx_;
};
}

#endif //CXXBASIC_POOLING_LAYER_H
