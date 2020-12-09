//
// Created by wyl on 2020/11/26.
//

#include "Blob.hpp"

namespace caffe{
    template <typename Dtype>
    void Blob<Dtype>::Reshape(const int num, const int channels, const int height, const int width) {
        vector<int> shape(4);
        shape[0] = num;
        shape[1] = channels;
        shape[2] = height;
        shape[3] = width;
        Reshape(shape);
    }

    template<typename Dtype>
    void Blob<Dtype>::Reshape(const vector<int> &shape) {
        count_ = 1;
        shape_.resize(shape.size());
        //shape vector<int>也可以放在gpu上
        if(!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)){
            shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));    // shape_data_智能指针指向刚生成的新的对象, 原先对象引用次数减1
        }
        int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
        for(int i=0; i < shape.size(); i++){
            count_ *= shape[i];
            shape_[i] = shape[i];
            shape_data[i] = shape[i];
        }
        // data_和diff_重置指向新的对象
        if(count_ > capacity_){
            capacity_ = count_;
            data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
            diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
        }
    }

    template <typename Dtype>
    void Blob<Dtype>::Reshape(const caffe::Blob<Dtype> &other) {
        Reshape(other.shape());
    }

    template <typename Dtype>
    const Dtype* Blob<Dtype>::cpu_data() const {
        return (const Dtype*) data_->cpu_data();
    }

    template <typename Dtype>
    const Dtype* Blob<Dtype>::gpu_data() const {
        CHECK(data_);
        return (const Dtype*) data_->gpu_data();
    }

    template <typename Dtype>
    const Dtype* Blob<Dtype>::cpu_diff() const {
        return (const Dtype*) diff_->cpu_data();
    }

    template <typename Dtype>
    const Dtype* Blob<Dtype>::gpu_diff() const {
        return (const Dtype*) diff_->gpu_data();
    }
    /**
     * @brief 如果传入的数据与原数据尺寸不一致，reset；如果一致，释放掉原先数据的内存，再指向新数据
     * @tparam Dtype
     * @param data
     */
    template <typename Dtype>
    void Blob<Dtype>::set_cpu_data(Dtype* data) {
        CHECK(data);
        size_t size = count_ * sizeof(Dtype);
        if(size != data_->size()){
            data_.reset(new SyncedMemory(size));
            diff_.reset(new SyncedMemory(size));
        }
        data_->set_cpu_data(data);
    }

    template <typename Dtype>
    void Blob<Dtype>::set_gpu_data(Dtype *data) {
        CHECK(data);
        size_t size = count_ * sizeof(Dtype);
        if(size!=data_->size()){
            data_.reset(new SyncedMemory(size));
            diff_.reset(new SyncedMemory(size));
        }
        data_->set_gpu_data(data);
    }

    /**]
     * @brief 取出cpu_ptr_
     * @tparam Dtype
     * @return static_cast强制类型转换
     */
    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_cpu_data() {
        return static_cast<Dtype*>(data_->mutable_cpu_data());
    }

    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_gpu_data() {
        return static_cast<Dtype*>(data_->mutable_gpu_data());
    }

    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_cpu_diff() {
        return static_cast<Dtype*>(diff_->mutable_cpu_data());
    }

    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_gpu_diff() {
        return static_cast<Dtype*>(diff_->mutable_cpu_data());
    }

    template <> void Blob<unsigned int>::Update() {NOT_IMPLEMENTED;}
    template <> void Blob<int>::Update() {NOT_IMPLEMENTED;}

    template <typename Dtype>
    void Blob<Dtype>::Update(){
        switch (data_->head()){
            case SyncedMemory::HEAD_AT_CPU:
                caffe_axpy<Dtype>(count_, Dtype(-1),
                                  static_cast<Dtype>(diff_->cpu_data()),
                                  static_cast<Dtype>(data_->mutable_cpu_data())
                                  );
                break;

            case SyncedMemory::HEAD_AT_GPU:
            case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
                check_device();
                caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
                                      static_cast<Dtype>(diff_->gpu_data()),
                                      static_cast<Dtype>(data_->mutable_gpu_data())
                                      );
#else
                NO_GPU;
#endif
                break;
            default:
                LOG(FATAL) << "Syncedmem not initialized.";
        }
    }

    template <> unsigned int Blob<unsigned int>::asum_data() const{
        NOT_IMPLEMENTED;
        return 0;
    }

    template <> int Blob<int>::asum_data() const{
        NOT_IMPLEMENTED;
        return 0;
    }

    template <typename Dtype>
    Dtype Blob<Dtype>::asum_data() const {
        switch(data_->head()){
            case SyncedMemory::HEAD_AT_CPU:
                return caffe_cpu_asum(count_, cpu_data());
                break;
            case SyncedMemory::HEAD_AT_GPU:
            case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    Dtype asum;
    caffe_gpu_asum(count_, gpu_data(), &asum);
    return asum;
#else
    NO_GPU;
#endif
            case SyncedMemory::UNINITIALIZED:
                return 0;
            default:
                LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
        }
        return 0;
    }

    template <typename Dtype>
    void Blob<Dtype>::ShareData(const Blob<Dtype> &other) {
        data_ = other.data();
    }

    template <typename Dtype>
    void Blob<Dtype>::ShareDiff(const Blob<Dtype> &other) {
        diff_ = other.data();
    }

    template <typename Dtype>
    void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
        if (reshape) {
            vector<int> shape;
            if (proto.has_num() || proto.has_channels() ||
                proto.has_height() || proto.has_width()) {
                // Using deprecated 4D Blob dimensions --
                // shape is (num, channels, height, width).
                shape.resize(4);
                shape[0] = proto.num();
                shape[1] = proto.channels();
                shape[2] = proto.height();
                shape[3] = proto.width();
            } else {
                shape.resize(proto.shape().dim_size());
                for (int i = 0; i < proto.shape().dim_size(); ++i) {
                    shape[i] = proto.shape().dim(i);
                }
            }
            Reshape(shape);
        } else {
            CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
        }
        // copy data
        Dtype* data_vec = mutable_cpu_data();
        if (proto.double_data_size() > 0) {
            CHECK_EQ(count_, proto.double_data_size());
            for (int i = 0; i < count_; ++i) {
                data_vec[i] = proto.double_data(i);
            }
        } else {
            CHECK_EQ(count_, proto.data_size());
            for (int i = 0; i < count_; ++i) {
                data_vec[i] = proto.data(i);
            }
        }
        if (proto.double_diff_size() > 0) {
            CHECK_EQ(count_, proto.double_diff_size());
            Dtype* diff_vec = mutable_cpu_diff();
            for (int i = 0; i < count_; ++i) {
                diff_vec[i] = proto.double_diff(i);
            }
        } else if (proto.diff_size() > 0) {
            CHECK_EQ(count_, proto.diff_size());
            Dtype* diff_vec = mutable_cpu_diff();
            for (int i = 0; i < count_; ++i) {
                diff_vec[i] = proto.diff(i);
            }
        }
    }

    template <>
    void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {
        proto->clear_shape();
        for (int i = 0; i < shape_.size(); ++i) {
            proto->mutable_shape()->add_dim(shape_[i]);
        }
        proto->clear_double_data();
        proto->clear_double_diff();
        const double* data_vec = cpu_data();
        for (int i = 0; i < count_; ++i) {
            proto->add_double_data(data_vec[i]);
        }
        if (write_diff) {
            const double* diff_vec = cpu_diff();
            for (int i = 0; i < count_; ++i) {
                proto->add_double_diff(diff_vec[i]);
            }
        }
    }

    template <>
    void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {
        proto->clear_shape();
        for (int i = 0; i < shape_.size(); ++i) {
            proto->mutable_shape()->add_dim(shape_[i]);
        }
        proto->clear_data();
        proto->clear_diff();
        const float* data_vec = cpu_data();
        for (int i = 0; i < count_; ++i) {
            proto->add_data(data_vec[i]);
        }
        if (write_diff) {
            const float* diff_vec = cpu_diff();
            for (int i = 0; i < count_; ++i) {
                proto->add_diff(diff_vec[i]);
            }
        }
    }

    INSTANTIATE_CLASS(Blob);
    template class Blob<int>;
    template class Blob<unsigned int>;
}