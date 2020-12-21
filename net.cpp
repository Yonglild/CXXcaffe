//
// Created by wyl on 2020/11/26.
//


// https://www.cnblogs.com/yymn/articles/7498516.html
// https://blog.csdn.net/gaoenyang760525/article/details/72874816
// https://zhuanlan.zhihu.com/p/81667754
// https://blog.csdn.net/qq_28660035/article/details/80365570
// https://blog.csdn.net/ricky5000/article/details/68930978
#include "include/net.hpp"
namespace caffe{
template<typename Dtype>
Net<Dtype>::Net(const NetParameter &param) {
    Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string &param_file, Phase phase) {

}

template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter &param, const int layer_id, const int bottom_id,
                             set <string> *available_blobs, map<string, int> *blob_name_to_idx) {

}


};