//
// Created by wyl on 2020/12/11.
//

#ifndef CXXBASIC_LAYER_HPP
#define CXXBASIC_LAYER_HPP
#include "include/proto/caffe.pb.h"
#include <vector>
#include "Blob.hpp"

using namespace std;
namespace caffe{
    template <typename Dtype>
    class Layer{
    public:
        explicit Layer(const LayerParameter param):layer_param_(param){
            //phase_设为Train/Test   param为存储在proto中的层的超参数
            phase_ = param.phase();
            if(layer_param_.blobs_size()>0){
                blobs_.resize(layer_param_.blobs_size());
                for(int i=0; i<layer_param_.blobs_size(); ++i){
                    blobs_[i].reset(new Blob<Dtype>());
                    blobs_[i].FromProto(layer_param_.blobs(i));
                }
            }
        }
        virtual ~Layer() {}

        void SetUp(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top){
            CheckBlobCounts(bottom, top);
            LayerSetUp(bottom, top);
            Reshape(bottom, top);
            SetLossWeights(top);
        }

        // top是未成型的，在Forward之前调用该函数调整top的形状
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){}

        //根据输入blob的形状，按照需要调整top blobs形状，并且调整内部缓存的形状
        //和其他必要的调整，使该层能适应bottom blobs
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){}

        inline Dtype Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

        inline void Backward(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        // 返回左值
        vector<shared_ptr<Blob<Dtype>>>& blobs(){
            return blobs_;
        }

        const LayerParameter& layer_param() {
            return layer_param_;
        }



    protected:
        LayerParameter layer_param_;
        Phase phase_;
        vector<shared_ptr<Blob<Dtype>>> blobs_;
        vector<bool> param_propagate_down_;
        vector<Dtype> loss_;

        // virtual fun()=0; 表示纯虚函数
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) = 0;
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) = 0;
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
            return Forward_cpu(bottom, top);
        };
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
            return Backward_cpu(top, propagate_down, bottom);
        };

    };
}

// inline关键字应该出现在函数的定义中
template <typename Dtype>
inline Dtype caffe::Layer<Dtype>::Forward(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
    Dtype loss = 0;
    Reshape(bottom, top);
    switch(Caffe::mode()){
        case Caffe::CPU:
            Forward_cpu(bottom, top);
            for(int topid=0; topid<top.size(); ++topid){
                if(!this->loss(topid)){continue;}
                const int count = top[topid]->count();
                const Dtype* data = top[topid]->cpu_data();
                const Dtype* loss_weights = top[topid]->cpu_diff();     // 梯度如何计算
                loss += caffe_cpu_dot(count, data, loss_weights);
            }
            break;
        case Caffe::GPU:
#ifndef CPU_ONLY
            Forward_gpu(bottom, top);
            for(int topid=0; topid<top.size(); ++topid){
                if(!this->loss(topid)){continue;}
                const int count = top[topid]->count();
                const Dtype* data = top[topid]->gpu_data();
                const Dtype* loss_weights = top[topid]->gpu_diff();
                loss += caffe_gpu_dot(count, data, loss_weights);
            }
#endif
            break;
        default:
            LOG(FATAL) << "Unknown caffe mode.";
    }
    return loss;
}

template <typename Dtype>
inline void caffe::Layer<Dtype>::Backward(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                          const vector<Blob<Dtype> *> &bottom) {
    switch(Caffe::mode()){
        case Caffe::CPU:
            Backward_cpu(top, propagate_down, bottom);
            break;
        case Caffe::GPU:
#ifndef CPU_ONLY
            Backward_gpu(top, propagate_down, bottom);
#endif
            break;
        default:
            LOG(FATAL)<<"Unknown caffe mode!";
    }
}





#endif //CXXBASIC_LAYER_HPP
