//
// Created by wyl on 2020/11/26.
//
// 类模板与模板类 https://blog.csdn.net/low5252/article/details/94654468
#ifndef CXXBASIC_NET_H
#define CXXBASIC_NET_H

#include <vector>
#include <glog/logging.h>       // /usr/include/glog/logging.h
#include "Blob.hpp"
#include "include/proto/caffe.pb.h"
#include "layer.hpp"

using namespace std;
namespace caffe {
    template<typename Dtype>
    class Net {
    public:
        explicit Net(const NetParameter &param);

        explicit Net(const string &param_file, Phase phase);

        virtual ~Net();

        void Init(const NetParameter &in_param);
        void FilterNet()
        const vector<Blob<Dtype> *> &Forward(Dtype *loss = NULL);

        const vector<Blob<Dtype> *> &ForwardPrefilled(Dtype *loss = NULL) {
            LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: ForwardPrefilled() "
                                       << "will be removed in a future version. Use Forward().";
            return Forward(loss);
        }

        Dtype ForwardFromTo(int start, int end);

        Dtype FrowardFrom(int start);

        Dtype FrowardTo(int end);

        const vector<Blob<Dtype> *> &Forward(const vector<Blob<Dtype> *> &bottom,
                                             Dtype *loss = NULL);

        void ClearParamDiffs();

        void Backward();

        void BackwardFromTo(int start, int end);

        void BackwardFrom(int start);

        void BackwardTo(int end);

        void Reshape();

        Dtype ForwardBackward() {
            Dtype loss;
            Forward(&loss);
            Backward();
            return loss;
        }

        void Update();

        void SharedWeights();
        static void FilterNet(const NetParameter& param,
                NetParameter* param_filtered);
    protected:
        int AppendBottom(const NetParameter& param, const int layer_id, const int bottom_id, set<string>* available_blobs, map<string, int>* blob_name_to_idx);
        int AppendTop(const NetParameter& param, const int layer_id, const int bottom_id, set<string>* available_blobs, map<string, int>* blob_name_to_idx);
        Phase phase_;
        string name_;
        vector<vector<Blob<Dtype>*>> bottom_vecs_;
        vector<vector<int>> bottom_id_vecs_;
        vector<vector<bool>> bottom_need_backward_;

        vector<vector<Blob<Dtype>*>> top_vecs_;
        vector<vector<int>> top_id_vecs_;

        vector<shared_ptr<Layer<Dtype>>> layers_;
        vector<string> layer_names_;
        vector<bool> blob_need_backward_;
        vector<shared_ptr<Blob<Dtype>>> blobs_;
        vector<int> net_input_blob_indices_;
        vector<int> net_output_blob_indices_;
        vector<Blob<Dtype>*> net_input_blobs_;
        vector<Blob<Dtype>*> net_output_blobs_;

    };
}

#endif  //CXXBASIC_NET_H