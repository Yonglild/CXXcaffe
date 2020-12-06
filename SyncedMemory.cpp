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

    inline void SyncedMemory::to_gpu(){
        check_device();
#ifndef CPU_ONLY
        switch(head_){
            case UNINITIALIZED:
                CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
                caffe_gpu_memset(size_, 0, gpu_ptr_);
                head_ = HEAD_AT_GPU;
                own_gpu_data_ = true;
                break;
            case HEAD_AT_CPU:
                if(gpu_ptr_ == NULL){
                    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
                    own_gpu_data_ = true;
                }
                caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
                head_ = HEAD_AT_CPU;
                break;
            case HEAD_AT_CPU:
            case SYNCED:
                break;
        }
#else
    NO_GPU
#endif
    }


    const void* SyncedMemory::cpu_data() {
        check_device();
        to_cpu();
        return(const void*) cpu_ptr_;
    }

    const void* SyncedMemory::gpu_data() {
        check_device();
        to_gpu();
        return (const void*) gpu_ptr_;
    }

    /**
     * @brief 如果数据是自建的，释放掉自建的数据;并指向外部给的数据data
     * @param data
     */
    void SyncedMemory::set_cpu_data(void *data) {
        check_device();
        CHECK(data);
        if(own_cpu_data_){
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_data_);
        }
        cpu_ptr_ = data;
        head_ = HEAD_AT_CPU;
        own_cpu_data_ = false;
    }

    void SyncedMemory::set_gpu_data(void *data) {
        check_device();
#ifndef CPU_ONLY
        CHECK(data);
        if(own_gpu_data_){
            CUDA_CHECK(cudaFree(gpu_ptr_));
        }
        gpu_ptr_ = data;
        own_gpu_data_ = false;
        head_ = HEAD_AT_GPU;
#else
        NO_GPU;
#endif
    }

    /**
     * @brief 返回更新的cpu数据
     * @return
     */
    void* SyncedMemory::mutable_cpu_data() {
        check_device();
        to_cpu();
        head_ = HEAD_AT_CPU;
        return cpu_ptr_;
    }

    void* SyncedMemory::mutable_gpu_data() {
        check_device();
        to_gpu();
        head_ = HEAD_AT_GPU;
        return gpu_ptr_;
    }
}