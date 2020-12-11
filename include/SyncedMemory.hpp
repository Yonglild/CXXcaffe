//
// Created by wyl on 2020/11/28.
//

#ifndef CXXBASIC_SYNCEDMEMORY_HPP
#define CXXBASIC_SYNCEDMEMORY_HPP
#include <iostream>
namespace caffe {
    inline void CaffeMallocHost(void **ptr, size_t size, bool *use_cuda) {
#ifndef CPU_ONLY
        if (Caffe::mode() == Caffe::GPU) {
            cudaMallocHost(ptr, size);
            *use_cucda = true;
            return;
        }
#endif
#ifdef USE_MKL
    *ptr = mkl_malloc(size ? size:1, 64);
#else
    *ptr = malloc(size);
#endif
    *use_cuda = false;
    }

    inline void CaffeFreeHost(void *ptr, bool use_cuda){
#ifndef CPU_ONLY
        if(use_cuda){
            CUDA_CHECK(cudaFreeHost(ptr));
            return;
        }
    }
#endif

    class SyncedMemory {
    public:
        SyncedMemory();
        explicit SyncedMemory(size_t size);
        ~SyncedMemory();
        const void* cpu_data();         // 返回分配的cpu的内存地址：cpu_ptr_
        void set_cpu_data(void* data);  // cpu_ptr_所指向的内存释放，并且cpu_ptr_指向入参data所指向内存
        const void* gpu_data();         // 返回分配的gpu的内存地址：gpu_ptr_
        void set_gpu_data(void* data);  // gpu_ptr_所指向的内存释放
        void* mutable_cpu_data();       // 返回cpu_ptr_
        void* mutable_gpu_data();
        enum SyncedHead{ UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED};
        SyncedHead head() const {return head_;}
        size_t size() const{return size_;};

    private:
        void check_device();
        void to_cpu();
        void to_gpu();
        void* cpu_ptr_;     // void* 能包容地接受任何类型的指针
        void* gpu_ptr_;
        SyncedHead head_;
        size_t size_;
        bool own_gpu_data_;
        bool own_cpu_data_;
        bool cpu_malloc_use_data_;
        int device_;
    };
}

#endif //CXXBASIC_SYNCEDMEMORY_HPP
