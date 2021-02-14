//
// Created by wyl on 21-2-9.
//

#ifndef CXXBASIC_MATH_FUNCTIONS_H
#define CXXBASIC_MATH_FUNCTIONS_H

namespace caffe{
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
        const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
        const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
        Dtype* C);
}


#endif //CXXBASIC_MATH_FUNCTIONS_H
