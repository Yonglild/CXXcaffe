//
// Created by wyl on 2020/11/26.
//
#ifndef CXXBASIC_BLOB_H
#define CXXBASIC_BLOB_H

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <memory>
#include "SyncedMemory.hpp"
using namespace std;

const int kMaxBlobAxes = 32;

/*
 * 指针的好处：
 * 通过函数改变一个变量的值
 * 数据量太大，用指针来做形参，提高效率
 * 灵活的数据类型转换
 */

namespace caffe{
    template <typename Dtype>
    class Blob{
    public:
        // 有三种构造函数
        Blob():data_(), diff_(), count_(0), capacity_(0){}
        explicit Blob(const int num, const int channels, const int height ,const int width);
        explicit Blob(const vector<int>& shape);

        // 有三种Reshape输入方式及ReshapeLike
        // 将data_, diff_, shape_data_所指向的对象都重置一遍
        void Reshape(const int num, const int channels, const int height, const int width);
        void Reshape(const vector<int>& shape);
//        void Reshape(const BlobShape& shape);
        void Reshape(const Blob& other);

        // 把返回值复制到外部临时的存储单元中，copied from shape_
        // const放在前面修饰返回值。shape_为const类型。
        inline const vector<int>& shape() const {
            return shape_;
        }

        inline const int shape(int axis) const{
            return shape_[axis];
        }

        inline string shape_string() const {
            std::stringstream oss;
            for(int i=0; i<shape_.size(); i++){
                oss << shape_[i] << " ";
            }
            oss << "(" << count_ << ")";
            return oss.str();
        }

        // blob的维数
        inline const int num_axes() const {
            return shape_.size();
        }

        // blob
        inline int count() const{
            return count_;
        }

        /**
         * @brief 获取从start_axis到end_axis的容量
         * @param start_axis, end_axis
         * @return
         */
        inline int count(int start_axis, int end_axis){
            CHECK_LE(start_axis, end_axis);
            CHECK_GE(start_axis, 0);
            CHECK_GE(end_axis, 0);
            CHECK_LE(start_axis, num_axes());
            CHECK_LE(end_axis, num_axes());
            int count = 1;
            for(int i=start_axis; i<end_axis; ++i){
                count *= shape(i);
            }
            return count;
        }

        //
        inline int count(int start_axis) const{
            return count(start_axis, num_axes());
        }

        /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
        inline int num() const { return LegacyShape(0); }
        /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
        inline int channels() const { return LegacyShape(1); }
        /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
        inline int height() const { return LegacyShape(2); }
        /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
        inline int width() const { return LegacyShape(3); }
        inline int LegacyShape(int index) const {
            CHECK_LE(num_axes(), 4)
                    << "Cannot use legacy accessors on Blobs with > 4 axes.";
            CHECK_LT(index, 4);
            CHECK_GE(index, -4);
            if (index >= num_axes() || index < -num_axes()) {
                // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
                // indexing) -- this special case simulates the one-padding used to fill
                // extraneous axes of legacy blobs.
                return 1;
            }
            return shape(index);
        }

        // indices对应的blob存放位置, 四维向量在内存中以一维形式存放
        // 行优先，列其次，c第三，n最后[(n*channel()+c)*height()+h]*width()+w]
        inline int offset(const vector<int>& indices) const{
            int offset = 0;
            for(int i=0; i<indices.size(); ++i){
                offset *= shape(i);
                if(indices.size() > i){
                    offset += indices[i];
                }
            }
            return offset;
        }

        inline int offset(const int n, const int c, const int h, const int w){
            return ((((n*channels()+c))*height()+h)*width()+w);
        }

        inline int CanonicalAxisIndex(int axis_index) const {
            CHECK_GE(axis_index, -num_axes())
                    << "axis " << axis_index << " out of range for " << num_axes()
                    << "-D Blob with shape " << shape_string();
            CHECK_LT(axis_index, num_axes())
                    << "axis " << axis_index << " out of range for " << num_axes()
                    << "-D Blob with shape " << shape_string();
            if (axis_index < 0) {
                return axis_index + num_axes();
            }
            return axis_index;
        }


        // 使用cudaMemcpy或者memcpy进行内存的拷贝
        void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false, bool reshape = false);

        // 返回blob中指定位置的数据
        // const 放在函数后面，函数内部变量不能改变
        // 返回的都是cpu上的数据
        inline Dtype data_at(const int n, const int c, const int h, const int w) const {
            return cpu_data()[offset(n, c, h, w)];
        }

        inline Dtype data_at(const vector<int>& index) const {
            return cpu_data()[offset(index)];
        }

        inline Dtype diff_at(const int n, const int c, const int h, const int w){
            return cpu_diff()[offset(n, c, h, w)];
        }

        inline Dtype diff_at(const vector<int>& index) const {
            return cpu_diff()[offset(index)];
        }

        inline const shared_ptr<SyncedMemory>& data() const{
            return data_;
        };

        inline const shared_ptr<SyncedMemory>& diff() const{
            return diff_;
        };

        const Dtype* cpu_data() const;      // 返回SyncedMemory类型中的cpu_ptr_
        void set_cpu_data(Dtype* data);     // 将data赋给blob中的data_
        const Dtype* gpu_data() const;
        void set_gpu_data(Dtype* data);
        const Dtype* cpu_diff() const;

        Dtype* mutable_cpu_data();
        Dtype* mutable_gpu_data();
        Dtype* mutable_cpu_diff();
        Dtype* mutable_gpu_diff();

        void Update();
        Dtype* asum_data() const;
        void ShareData(const Blob& other);
        void ShareDiff(const Blob& other);
        void FromProto(const BlobProto& proto, bool reshape = true);
        void ToProto(BlobProto* proto, bool write_diff = false) const;

    protected:
        //Blob中有三种数据需要GPU和CPU同步
        shared_ptr<SyncedMemory> data_;
        shared_ptr<SyncedMemory> diff_;
        shared_ptr<SyncedMemory> shape_data_;       // SyncedMemory CPU与GPU数据同步
        vector<int> shape_;
        int count_;             // 张量中元素的数量（nchw）
        int capacity_;          // 分配的内存大小，reshape的时候可能需要扩容
    };
}

#endif //CXXBASIC_BLOB_H
