//
// Created by wyl on 21-2-10.
//
namespace caffe{
    // 模板函数
// 模板特化 直接为某个特定类型做特化，如特化为float, double等
template <>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
        const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
        const float alpha, const float* A,  const float* B, const float beta,
        float* C){
        int lda = (TransA == CblasNoTrans) ? K : M;
        int ldb = (TransB == CblasNoTrans) ? N : K;
        cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
                    ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
                            const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                            const double alpha, const double* A, const double* B, const double beta,
                            double* C) {
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
                ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
                           const int N, const float alpha, const float* A, const float* x,
                           const float beta, float* y) {
    cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
                            const int N, const double alpha, const double* A, const double* x,
                            const double beta, double* y) {
    cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}
}

