#include <algorithm>
#include <vector>

#include "../include/base_conv_layer.hpp"

namespace caffe{
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                             const vector<Blob<Dtype> *> &top) {
    // 设置卷积核的大小，补０，步长等
    // 根据protobuf中的层参数设置，配置卷积核的大小，padding，步长和输入等等。
    ConvolutionParameter conv_param = this->layer_param().convolution_param();
    force_nd_im2col_ = conv_param.force_nd_im2col(); //根据层参数设置是否强制进行n维im2col
    channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());    // 通道层在第几维，一般是２

}
}
