//
// Created by wyl on 21-1-28.
//

#ifndef CXXBASIC_IM2COL_HPP
#define CXXBASIC_IM2COL_HPP

namespace caffe{
template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
            const int height, const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w, const int stride_h,
            const int stride_w, const int dilation_h, const int dilation_w,
            Dtype* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const int dilation_h, const int dilation_w,
                Dtype* data_im);


}


#endif //CXXBASIC_IM2COL_HPP
