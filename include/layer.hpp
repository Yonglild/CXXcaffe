//
// Created by wyl on 2020/12/11.
//

#ifndef CXXBASIC_LAYER_H
#define CXXBASIC_LAYER_H
#include "include/proto/caffe.pb.h"
namespace caffe{

    template <typename Dtype>
    class Layer{
    public:
        explicit Layer(const LayerParameter& param)
        :layer_param_(param){

        }
    };
}


#endif //CXXBASIC_LAYER_H
