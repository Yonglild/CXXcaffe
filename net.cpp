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
void Net<Dtype>::Init(const NetParameter &in_param) {
    phase_ = in_param.state().phase();
    NetParameter filtered_param;
    FilterNet(in_param, &filtered_param);

    NetParameter param;
    InsertSplits(filtered_param, &param);

    // 建立所有层
    name_ = param.name();
    map<string, int> blob_name_to_idx;
    set<string> available_blobs;
    memory_used_ = 0;

    //设置每层的输入输出
    bottom_vecs_.resize(param.layer_size());
    top_vecs_.resize(param.layer_size());
    bottom_id_vecs_.resize(param.layer_size());
    top_id_vecs_.resize(param.layer_size());
    param_id_vecs_.resize(param.layer_size());  // 权重blob的id
    bottom_need_backward_.resize(param.layer_size());   // blob是否需要反向传播

    for(int layer_id=0; layer_id<param.layer_size(); layer_id++){
        // 如果没有设置层的phase，就从网络的phase继承，因为能通过FilterNet留下来肯定符合该phase
        if(!param.layer(layer_id).has_phase()){
            param.mutable_layer(layer_id)->set_phase(phase_);
        }

        // 设置层参数
        const LayerParameter& layer_param = param.layer(layer_id);
        if(layer_param.propagate_down_size()>0){
            CHECK_EQ(layer_param.propagate_down_size(),
                     layer_param.bottom_size())
                    << "propagate_down param must be specified "
                    << "either 0 or bottom_size times ";
        }
        // LayerRegistry?????????
        /*
        *把当前层的参数转换为shared_ptr<Layer<Dtype>>，
        *创建一个具体的层，并压入到layers_中
        */
        layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
        layer_names_.push_back(layer_param.name());
        LOG_IF(INFO, Caffe::root_solver())
                << "Creating Layer " << layer_param.name();
        bool need_backward = false;

        // bottom_size()？？？？
        for(int bottom_id = 0; bottom_id<layer_param.bottom_size(); ++bottom_id){
            const int blob_id = AppendBottom(param, layer_id, bottom_id,
                    &available_blobs, &blob_name_to_idx);
            need_backward |= blob_need_backward_[blob_id];
        }

        int num_top = layer_param.top_size();
        for(int top_id = 0; top_id<num_top; ++top_id){
            //通过AppendTop和AppendBottom, bottom_vecs_和top_vecs_连接在了一起
            //在AppendTop中会往available_blobs添加某层的输出blob,在AppendBottom中会
            //从available_blobs中删除前一层的输出blob，所有layers遍历完后剩下的就
            //是整个net的输出blob
            AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
            if(layer_param.type() == "Input"){
                const int blob_id = blobs_.size() - 1;
                net_input_blob_indices_.push_back(blob_id);
                net_input_blobs_.push_back(blobs_[blob_id].get());
            }
        }
    }





}

template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter &param, const int layer_id, const int bottom_id,
                             set <string> *available_blobs, map<string, int> *blob_name_to_idx) {

}


};