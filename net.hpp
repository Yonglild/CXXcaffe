//
// Created by wyl on 2020/11/26.
//
// 类模板与模板类 https://blog.csdn.net/low5252/article/details/94654468
#ifndef CXXBASIC_NET_H
#define CXXBASIC_NET_H

#include <vector>
#include <glog/logging.h>       // /usr/include/glog/logging.h
#include "Blob.hpp"

using namespace std;
namespace caffe {
    template<typename Dtype>
    class Net {
    public:
        explicit Net(const NetParameter &param);

        explicit Net(const string &param_file, Phase phase);

        virtual ~Net();

        void Init(const NetParameter &param);

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


    };
}

#endif  //CXXBASIC_NET_H