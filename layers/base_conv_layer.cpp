#include <algorithm>
#include <vector>

#include "../include/layers/base_conv_layer.hpp"
#include "../include/util/math_functions.hpp"
namespace caffe {
    template<typename Dtype>
    void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                 const vector<Blob<Dtype> *> &top) {
        // 设置卷积核的大小，补０，步长等
        // 根据protobuf中的层参数设置，配置卷积核的大小，padding，步长和输入等等。
        ConvolutionParameter conv_param = this->layer_param().convolution_param();
        force_nd_im2col_ = conv_param.force_nd_im2col(); //根据层参数设置是否强制进行n维im2col
        channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());    // 通道层在第几维
        const int first_spatial_axis = channel_axis_ + 1;
        const int num_axes = bottom[0]->num_axes();
        num_spatial_axes_ = num_axes - first_spatial_axis;  // 是２维的图像
        CHECK_GE(num_spatial_axes_, 0);

        // vector初始化，一个元素，值为num_spatial_axes_
        // 当num_spatial_axes_==2时，spatial_dim_blob_shape这个vector只包含一个元素且值为2
        vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));  // 为什么只包含一个元素

        // 调用blob.cpp里的 void Blob<Dtype>::Reshape(const vector<int>& shape)
        //以spatial_dim_blob_shape为参数来构造一个Blob，即kernel_shape_，则这个Blob的维度信息只包含一个维度，值为2,
        //也就是说这个Blob的count_==2。尽管这个Blob的维度信息只包含一个维度,只有两个数。
        //因为在后续的计算（Im2col）中，我只关心这个Blob中的数据的值，而不关心这个Blob的shape信息.
        //例如在Im2col()中，只要取出相应数值即可kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1]。
        kernel_shape_.Reshape(spatial_dim_blob_shape);  // ?????
        int *kernel_shape_data = kernel_shape_.mutable_cpu_data();  // 取出[h, w]
        if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
            CHECK_EQ(num_spatial_axes_, 2)
                    << "kernel_h & kernel_w can only be used for 2D convolution.";
            CHECK_EQ(0, conv_param.kernel_size_size())
                    << "Either kernel_size or kernel_h/w should be specified; not both.";
            kernel_shape_data[0] = conv_param.kernel_h();
            kernel_shape_data[1] = conv_param.kernel_w();
        } else {
            // 若层参数中没有定义卷积核宽和高，则根据卷积核的维度数来确定。哪一维核大小就附几
            // repeated uint32 kernel_size
            const int num_kernel_dims = conv_param.kernel_size_size();
            CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
                    << "kernel_size must be specified once, or once per spatial dimension "
                    << "(kernel_size specified " << num_kernel_dims << " times; "
                    << num_spatial_axes_ << " spatial dims).";
            for (int i = 0; i < num_spatial_axes_; i++) {
                kernel_shape_data[i] =
                        conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
            }
        }
        // 核实　卷积核每一个维度不能为0
        for (int i = 0; i < num_spatial_axes_; i++) {
            CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
        }

        // 接下来的stride_，pad_，dilation_这些blob的设定也类似。
        // Setup stride dimensions (stride_).
        stride_.Reshape(spatial_dim_blob_shape);
        int *stride_data = stride_.mutable_cpu_data();
        if (conv_param.has_stride_h() || conv_param.has_kernel_w()) {
            CHECK_EQ(num_spatial_axes_, 2)
                    << "stride_h & stride_w can only be used for 2D convolution.";
            CHECK_EQ(0, conv_param.stride_size())
                    << "Either stride or stride_h/w should be specified; not both.";
            stride_data[0] = conv_param.stride_h();
            stride_data[1] = conv_param.stride_w();
        } else {
            // repeated uint32 stride
            const int num_stride_dims = conv_param.stride_size();
            CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
                  num_stride_dims == num_spatial_axes_)
                    << "stride must be specified once, or once per spatial dimension "
                    << "(stride specified " << num_stride_dims << " times; "
                    << num_spatial_axes_ << " spatial dims).";
            const int kDefaultStride = 1;
            for (int i = 0; i < num_spatial_axes_; i++) {
                stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
                                 (num_stride_dims == 1) ? 0 : i;
                CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
            }
        }

        // Setup pad dimensions (pad_).
        pad_.Reshape(spatial_dim_blob_shape);
        int* pad_data = pad_.mutable_cpu_data();
        if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
            CHECK_EQ(num_spatial_axes_, 2)
                    << "pad_h & pad_w can only be used for 2D convolution.";
            CHECK_EQ(0, conv_param.pad_size())
                    << "Either pad or pad_h/w should be specified; not both.";
            pad_data[0] = conv_param.pad_h();
            pad_data[1] = conv_param.pad_w();
        } else {
            const int num_pad_dims = conv_param.pad_size();
            CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
                  num_pad_dims == num_spatial_axes_)
                    << "pad must be specified once, or once per spatial dimension "
                    << "(pad specified " << num_pad_dims << " times; "
                    << num_spatial_axes_ << " spatial dims).";
            const int kDefaultPad = 0;
            for (int i = 0; i < num_spatial_axes_; ++i) {
                pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
                              conv_param.pad((num_pad_dims == 1) ? 0 : i);
            }
        }

        // Setup dilation dimensions (dilation_).
        dilation_.Reshape(spatial_dim_blob_shape);
        int* dilation_data = dilation_.mutable_cpu_data();
        const int num_dilation_dims = conv_param.dilation_size();
        CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
              num_dilation_dims == num_spatial_axes_)
                << "dilation must be specified once, or once per spatial dimension "
                << "(dilation specified " << num_dilation_dims << " times; "
                << num_spatial_axes_ << " spatial dims).";
        const int kDefaultDilation = 1;
        for (int i = 0; i < num_spatial_axes_; ++i) {
            dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                               conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
        }

        // Special case: im2col is the identity for 1x1 convolution with stride 1
        // and no padding, so flag for skipping the buffer and transformation.
        // ？？？？？？？？？？？？？
        is_1x1_ = true;
        for (int i = 0; i < num_spatial_axes_; ++i) {
            is_1x1_ &=
                    kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
            if (!is_1x1_) { break; }
        }

        channels_ = bottom[0]->shape(channel_axis_);
        num_output_ = this->layer_param_.convolution_param().num_output();
        CHECK_GT(num_output_, 0);       // 大于
        group_ = this->layer_param_.convolution_param().group();
        CHECK_EQ(channels_%group_, 0);  // 等于
        if(reverse_dimensions()){
            conv_out_channels_ = channels_;
            conv_in_channels_ = num_output_;
        }else{
            conv_out_channels_ = num_output_;
            conv_in_channels_ = channels_;
        }

        vector<int> weight_shape(2);
        weight_shape[0] = conv_out_channels_;
        weight_shape[1] = conv_in_channels_ / group_;

        for(int i=0; i<num_spatial_axes_; ++i){
            weight_shape.push_back(kernel_shape_data[i]);
        }

        // 是否拥有bias_term
        bias_term_ = this->layer_param_.convolution_param().bias_term();
        vector<int> bias_shape(bias_term_, num_output_);
        if(this->blobs_.size()>0){
            // 已有权重，跳过初始化；否则需要初始化
            CHECK_EQ(1+bias_term_, this->blobs_.size())
            << "Incorrect number of weight blobs";
            if(weight_shape != this->blobs_[0].shape()){
                Blob<Dtype> weight_shaped_blob(weight_shape);
                LOG(FATAL)<<"Incorrect weight shape: expected shape "
                << weight_shaped_blob.shape_string() << "; instead, shape was"
                << this->blobs_.shape_string();
            }
            if(bias_term_ && bias_shape != this->blobs_[1].shape()){
                Blob<Dtype> bias_shaped_blob(bias_shape);
                LOG(FATAL) << "Incorrect bias shape: expected shape "
                           << bias_shaped_blob.shape_string() << "; instead, shape was "
                           << this->blobs_[1]->shape_string();
            }
            LOG(INFO) << "Skipping parameter initialization";
        }else{
            if(bias_term_){
                this->blobs_.resize(2);
            }else{
                this->blobs_.resize(1);
            }
            this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
            // ?????
            shared_ptr<Fillter<Dtype>> weight_fillter(GetFillter<Dtype>(
                    this->layer_param_.convolution_param().weight_filter()));
            weight_fillter->Fill(this->blobs_[0].get());

            if(bias_term_){
                this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
                shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                        this->layer_param_.convolution_param().bias_filler()));
                bias_filler->Fill(this->blobs_[1].get());
            }
        }
        kernel_dim_ = this->blobs_[0]->count(1);    //　滤波器大小：通道数 × h * w
        weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
        this->param_propagate_down_.resize(this->blobs_.size(), true);  // 滤波器默认可后向传播
}

// 继承自Layer
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
    const int first_spatial_axis = channel_axis_ + 1;
    CHECK_EQ(bottom[0]->num_axis(), first_spatial_axis + num_spatial_axes_)
    << "bottom num_axis may not change.";
    CHECK_EQ(bottom[0]->shape(channel_axis_), channel_axis_)
    << "Input size incompatible with convolution kernel.";
    // 如果输入多个blob，blob的尺寸要一致
    for(int bottom_id=1; bottom_id<bottom.size(); ++bottom_id){
        CHECK_EQ(bottom[0].shape()==bottom[bottom_id]->shape())
        << "shape mismatch - bottom[0]:" << bottom[0]->shape_string()
        << "vs. bottom[" << bottom_id << "]: "
        << bottom[bottom_id]->shape_string();
    }

    bottom_shape_ = &bottom[0]->shape(); // (b,c,h,w)
    // 虚函数，在继承类中具体实现。根据步长、padding等计算输出blob的尺寸
    compute_output_shape();     // 计算结果放哪里, 放在output_shape_???
    // vector初始化 [begin, end)左闭右开；
    vector<int> top_shape(bottom[0]->shape().begin(),
            bottom[0]->shape().begin() + channel_axis_);
    top_shape.push_back(num_output_);   // 输出个数
    // 将output_shape_压入top_shape
    for(int i=0; i<num_spatial_axes_; ++i){
        top_shape.push_back(output_shape_[i]);
    }
    // 重置blob
    for(int top_id=0; top_id<top.size(); top_id++){
        top[top_id]->Reshape(top_shape);
    }

    // 输出张量的单个通道的像素个数
    if(reverse_dimensions()){
        conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
    }else{
        conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
    }

    // 类似 kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_;
    col_offset_ = kernel_dim_ * conv_out_spatial_dim_;  // Cin矩阵的大小 （输入图像的计算矩阵）
    output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;   // Cout矩阵的大小

    // 输入图像的shape
    vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
    conv_input_shape_.Reshape(bottom_dim_blob_shape);   // [输入图像通道数, 输入图像h, 输入图像w]
    int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
    for(int i=0; i<num_spatial_axes_; i++){
        if(reverse_dimensions()){
            conv_input_shape_data[i] = top[0]->shape(channel_axis_ + 1);
        }else{
            conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + 1);
        }
    }

    // 得到col_buffer的形状
    // https://www.cnblogs.com/pursuiting/p/8563246.html
    // https://blog.csdn.net/u010417185/article/details/52192126
    // https://blog.csdn.net/ayst123/article/details/43924151
    // https://www.cnblogs.com/darkknightzh/p/10486686.html#_label3
    // col_buffer_shape_ [kernel_dim_, conv_out_spatial_dim_ ]
    col_buffer_shape_.clear();
    col_buffer_shape_.push_back(kernel_dim_ * group_);
    for(int i=0; i<num_spatial_axes_; i++){
        if(reverse_dimensions()){
            col_buffer_shape_.push_back(input_shape(i+1));
        }else{
            col_buffer_shape_.push_back(output_shape_[i]);
        }
    }
    // 将col_buffer变形
    col_buffer_.Reshape(col_buffer_shape_);
    bottom_dim_ = bottom[0]->count(channel_axis_);// 输入通道数*输入图像的h*输入图像的w
    top_dim_ = top[0]->count(channel_axis_);
    num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
    num_kernels_col2im_ = reverse_dimensions()?top_dim_:bottom_dim_;

    out_spatial_dim_ = top[0]->count(first_spatial_axis);
        if (bias_term_) {
            vector<int> bias_multiplier_shape(1, out_spatial_dim_);
            bias_multiplier_.Reshape(bias_multiplier_shape);
            caffe_set(bias_multiplier_.count(), Dtype(1),
                      bias_multiplier_.mutable_cpu_data());
        }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype *input, const Dtype *weights, Dtype *output,
                                                   bool skip_im2col) {
    const Dtype* col_buff = input;   // 一维数组
    if(!is_1x1_){
        if(!skip_im2col){
            conv_im2col_cpu(input, col_buffer_.mutable_cpu_data()); // col_buffer_是Blob类型
        }
        col_buff = col_buffer_.cpu_data();
    }
    // W*Cin = Cout 矩阵乘法
    for(int g = 0; g < group_; g++){
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_/group_,
                conv_out_spatial_dim_, kernel_dim_, (Dtype)1.,
                weights + weight_offset_ * g, col_buff + col_offset_ * g,
                (Dtype)0., output + output_offset_ * g);
    }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype *output, const Dtype *bias) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
            out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
                          (Dtype)1., output);
}
}
