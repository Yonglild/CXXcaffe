//
// Created by wyl on 21-2-14.
//

#include <vector>

#include "../include/layers/pooling_layer.hpp"

namespace caffe{
/**
 * @brief 从param中获取pad,stride,PoolMode等
 */
template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<caffe::Blob<Dtype> *> &bottom,
                                     const vector<caffe::Blob<Dtype> *> &top) {

}

/**
 * @brief Reshape top
 */
template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<caffe::Blob<Dtype> *> &bottom,
                                  const vector<caffe::Blob<Dtype> *> &top) {

}

/**
 * @brief 前向推理
 */
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<caffe::Blob<Dtype> *> &bottom,
                                      const vector<caffe::Blob<Dtype> *> &top) {

}

}

