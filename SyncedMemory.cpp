//
// Created by wyl on 2020/11/28.
//
// https://www.cnblogs.com/shine-lee/p/10050067.html
#include "SyncedMemory.hpp"
namespace caffe {
    SyncedMemory::SyncedMemory()
    :cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_data_(false), own_gpu_data_(false){
    }

    SyncedMemory::~SyncedMemory() {
        check_device();
        if(cpu_ptr_ && own_cpu_data_){
            CaffeFreeHost();        // CPU清除内存
        }
#ifndef CPU_ONLY
        if(gpu_ptr_ && own_gpu_data_){
            CUDA_CHECK(cudaFree(gpu_ptr_));
        }
#endif
    }

    inline void SyncedMemory::to_cpu() {
        check_device();
        switch(head_){
            case UNINITIALIZED:
                CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_data_);
            case HEAD_AT_GPU:
        #ifndef CPU_ONLY
            if(cpu_ptr_ == NULL){
                CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_data_);
                own_cpu_data_ = true;
            }
        #endif
        caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
            head_ = SYNCED;
            case HEAD_AT_CPU:
            case SYNCED:
        }
    }


}