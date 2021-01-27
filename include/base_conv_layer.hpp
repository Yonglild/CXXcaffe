//
// Created by wyl on 2021/1/8.
//
#ifndef CXXBASIC_BASE_CONV_LAYER_H
#define CXXBASIC_BASE_CONV_LAYER_H

#include "../include/Blob.hpp"
#include "../include/layer.hpp"
#include "../include/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
    template <typename Dtype>
    class BaseConvolutionLayer : public Layer<Dtype> {
    public:
        explicit BaseConvolutionLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline int MinBottomBlobs() const { return 1; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline bool EqualNumBottomTopBlobs() const { return true; }

    protected:
        // Helper functions that abstract away the column buffer and gemm arguments.
        // The last argument in forward_cpu_gemm is so that we can skip the im2col if
        // we just called weight_cpu_gemm with the same input.
        void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
                              Dtype* output, bool skip_im2col = false);
        void forward_cpu_bias(Dtype* output, const Dtype* bias);
        void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
                               Dtype* output);
        void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
        weights);
        void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
        void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
                              Dtype* output, bool skip_im2col = false);
        void forward_gpu_bias(Dtype* output, const Dtype* bias);
        void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
                               Dtype* col_output);
        void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
        weights);
        void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

        /// @brief The spatial dimensions of the input.输入的空间维度。
        inline int input_shape(int i) {
            return (*bottom_shape_)[channel_axis_ + i];
        }
        // reverse_dimensions should return true iff we are implementing deconv, so
        // that conv helpers know which dimensions are which.
        virtual bool reverse_dimensions() = 0;
        // Compute height_out_ and width_out_ from other parameters.
        virtual void compute_output_shape() = 0;

        /// @brief The spatial dimensions of a filter kernel.
        // 卷积核的形状[kernel_h, kernel_w]
        Blob<int> kernel_shape_;
        /// @brief The spatial dimensions of the stride.
        // 步长形状[stride_h, stride_w]
        Blob<int> stride_;
        /// @brief The spatial dimensions of the padding.
        // padding形状[pad_h, pad_w]
        Blob<int> pad_;
        /// @brief The spatial dimensions of the dilation.
        // 扩张卷积的形状，就是镂空式的卷积
        Blob<int> dilation_;
        /// @brief The spatial dimensions of the convolution input.
        // 卷积的输入形状 = [输入图像通道数, 输入图像h, 输入图像w]
        Blob<int> conv_input_shape_;
        /// @brief The spatial dimensions of the col_buffer.
        // col_buffer的形状 = [kernel_dim_, conv_out_spatial_dim_ ]
        // 即将输入图像转化成利于卷积的展开体col形式（具体参考src/util/im2col.cpp），存于col_buffer.
        vector<int> col_buffer_shape_;
        /// @brief The spatial dimensions of the output.
        // 输出blob的空间形状，一般是二维，存在vector里
        vector<int> output_shape_;
        // 层输入的形状，存在vector里，返回指针，因为是别的层的输出，直接用指针指向之前已经存在的上一层的output_shape_。
        const vector<int>* bottom_shape_;
        // 空间轴个数，就是输入是几维图像,一般为２维
        int num_spatial_axes_;
        // 输入度维度 = 输入通道数*输入图像的h*输入图像的w
        int bottom_dim_;
        // 输出维度 = 输出通道数*输出图像的h*输出图像的w
        int top_dim_;
        int channel_axis_;  // 输入图像的哪个axis是channel,一般是第二个维度
        // 堆大小
        int num_;
        // 通道数
        int channels_;
        // 卷积组的大小
        int group_;
        // 输出空间维度 = 卷积之后的图像长*卷积之后图像的宽
        int out_spatial_dim_;
        // 使用卷积组用到的权值偏置
        int weight_offset_;
        // 卷积后的图像的通道数
        int num_output_;
        // 是否启用偏置
        bool bias_term_;
        // 是否是1x1卷积
        bool is_1x1_;
        // 强制使用n维通用卷积，即im2col的n维形式，而不是更常用的二维形式。
        bool force_nd_im2col_;

    private:
        // wrap im2col/col2im so we don't have to remember the (long) argument lists
        // 将im2col/col2im封装，就不用自己来输入这么长的输入列表了。
        //void im2col_cpu(const Dtype* data_im, const int channels,
        //const int height, const int width, const int kernel_h, const int kernel_w,
        //const int pad_h, const int pad_w,
        //const int stride_h, const int stride_w,
        //const int dilation_h, const int dilation_w,
        //Dtype* data_col)
        inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
            if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
                im2col_cpu(data, conv_in_channels_,
                           conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                           kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                           pad_.cpu_data()[0], pad_.cpu_data()[1],
                           stride_.cpu_data()[0], stride_.cpu_data()[1],
                           dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
            } else {
                im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
                              col_buffer_shape_.data(), kernel_shape_.cpu_data(),
                              pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
            }
        }
        inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
            if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
                col2im_cpu(col_buff, conv_in_channels_,
                           conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                           kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                           pad_.cpu_data()[0], pad_.cpu_data()[1],
                           stride_.cpu_data()[0], stride_.cpu_data()[1],
                           dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
            } else {
                col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
                              col_buffer_shape_.data(), kernel_shape_.cpu_data(),
                              pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
            }
        }
#ifndef CPU_ONLY
        inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
            if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
                im2col_gpu(data, conv_in_channels_,
                           conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                           kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                           pad_.cpu_data()[0], pad_.cpu_data()[1],
                           stride_.cpu_data()[0], stride_.cpu_data()[1],
                           dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
            } else {
                im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
                              conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
                              kernel_shape_.gpu_data(), pad_.gpu_data(),
                              stride_.gpu_data(), dilation_.gpu_data(), col_buff);
            }
        }
        inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
            if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
                col2im_gpu(col_buff, conv_in_channels_,
                           conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                           kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                           pad_.cpu_data()[0], pad_.cpu_data()[1],
                           stride_.cpu_data()[0], stride_.cpu_data()[1],
                           dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
            } else {
                col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
                              conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
                              kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
                              dilation_.gpu_data(), data);
            }
        }
#endif

        int num_kernels_im2col_;
        int num_kernels_col2im_;
        int conv_out_channels_;
        int conv_in_channels_;
        int conv_out_spatial_dim_;  // 输出张量的单个通道的像素个数
        int kernel_dim_;
        int col_offset_;            // Cin矩阵的offset(排成一行的元素总数)
        int output_offset_;

        Blob<Dtype> col_buffer_;
        Blob<Dtype> bias_multiplier_;
    };

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#endif //CXXBASIC_BASE_CONV_LAYER_H
