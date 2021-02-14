//
// Created by wyl on 21-2-14.
//

#ifndef CXXBASIC_CONV_LAYER_H
#define CXXBASIC_CONV_LAYER_H

#include <vector>
#include "../Blob.hpp"
#include "../layer.hpp"
#include "../proto/caffe.pb.h"
#include "../layers/base_conv_layer.hpp"

namespace caffe{
    template <typename Dtype>
    class ConvolutionLayer:public BaseConvolutionLayer<Dtype>{
    public:
        explicit ConvolutionLayer(const LayerParameter& param)
        : BaseConvolutionLayer<Dtype>(param){}

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void compute_output_shape();
    };
}


#endif //CXXBASIC_CONV_LAYER_H
