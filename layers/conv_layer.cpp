//
// Created by wyl on 2021/1/8.
//
#include <vector>
#include "../include/layers/conv_layer.hpp"

namespace caffe{


template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {

}
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<caffe::Blob<Dtype> *> &bottom,
                                          const vector<caffe::Blob<Dtype> *> &top) {
    const Dtype* weight = this->blobs_[0]->cpu_data();
    for(int i = 0; i < bottom.size(); i++){
        const Dtype* bottom_data = this->blobs_[0]->cpu_data();
        Dtype* top_data = top[i]->mutable_cpu_data();
        for(int n = 0; n < this->num_; n++){
            this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                    top_data + n * this->top_dim_);
            if(this->bias_term_){
                const Dtype* bias = this->blobs_[1]->cpu_data();
                this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
            }
        }
    }
}


}
