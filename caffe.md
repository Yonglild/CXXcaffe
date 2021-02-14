blob类说明 https://www.cnblogs.com/ymjyqsx/p/7799731.html

**data_、diff_、shape_data_都是数据类型为syncedMemory的智能指针**

正是因为智能指针可以自动控制内存的释放，所以blob居然没有析构函数。

## SyncedMemory类 内存管理机制

https://www.cnblogs.com/shine-lee/p/10050067.html

负责caffe底层的在GPU和CPU的内存管理（创建，释放，GPU和CPU数据同步）

如果有gpu则使用CUDA内置函数：

cpu内存：CaffeMallocHost，CaffeFreeHost

gpu内存：cudaMalloc，cudaFree

```cpp
// Assuming that data are on the CPU initially, and we have a blob.
const Dtype* foo;
Dtype* bar;
foo = blob.gpu_data(); // data copied cpu->gpu.
foo = blob.cpu_data(); // no data copied since both have up-to-date contents.
bar = blob.mutable_gpu_data(); // no data copied.
// ... some operations ...
bar = blob.mutable_gpu_data(); // no data copied when we are still on GPU.
foo = blob.cpu_data(); // data copied gpu->cpu, since the gpu side has modified the data
foo = blob.gpu_data(); // no data copied since both have up-to-date contents
bar = blob.mutable_cpu_data(); // still no data copied.
bar = blob.mutable_gpu_data(); // data copied cpu->gpu.
bar = blob.mutable_cpu_data(); // data copied gpu->cpu.
```

## 2020/12/11 layer

## 2020/12/14 Google Protocol Buffer

g++ 基本使用

g++ 四个阶段：预处理，编译器，汇编器，链接器

```bash
g++  -c file.cpp   #-c   compile  生成file.o文件
g++ *.cpp -o file  # -o  指定输出文件名
g++ -l
```

## 2021/01/03 Net
https://www.cnblogs.com/xiangfeidemengzhu/p/7100440.html

blobs_ 存储整个Net的层之间中间计算结果（blob智能指针）
（Layer中的blobs_ 存储该层的可学习参数）

blob_id 就是在blobs_中的id

blob_names_ 就是blobs_中的name

blob_need_backward_　就是blobs_中的是否反向传播，默认为false

blob_name_to_idx 存储整个blobs_的blob_id和name


```cpp
  const string& blob_name = layer_param.bottom(bottom_id); // 通过bottom获取blob_name
```

**AppenBottom**

```
往第layer_id层的bottom_id位置设置输入向量，则availabel_blobs删掉该向量；设置该向量是否反传(默认false);返回该向量在blobs_中的位置
```
**AppenParam**

- 将第layer_id层param_id参数压入params_
- 根据判断该参数是第一次出现还是多次出现，如果多次则认为共享参数。获取参数的lr和decay信息
- 共享参数，必须有lr和decay????

| vector<shared_ptr<Blob<Dtype>>> params      | 存储整个Net的参数，其中包括各Layer中的blobs_               |
| ------------------------------------------- | ---------------------------------------------------------- |
| vector<vector<int>> param_id_vecs_          | 存储各层param的id，id指在整个网络params_中的位置           |
| vector<string> param_display_names_         |                                                            |
| vector<pair<int, int>> param_layer_indices_ | 将layer_id, param_id打包。这里param_id指在某一层中的局部id |

        learnable_params_, has_params_lr_, has_params_decay_, params_lr_, params_weight_decay_ 维度相同
        learnable_params_是否与params_维度相同???


## 2021/01/08 base_conv_layer

base_conv_layer继承自Layer

**LayerSetUp**

1. 根据conv_param设置kernel_shape_, stride_, pad, dilation  等blob<int>（尺寸）；
2. 根据输入的张量和conv_param设置滤波器的尺寸weight_shape，核实尺寸是否与已有滤波器相同；若滤波器不存在，则初始化滤波器；
3. 设置滤波器参数反传播

**Reshape**

继承自Layer中的Reshape()

1. 核实bottom的尺寸
2. 计算输出张量的尺寸，并初始化输出张量(重置)
3. 

## 2021/02/27 im2col.cpp
im2col  得到计算矩阵data_col，data_col本质上是一维的数组。
本质上图像并没有填充0,只是用了一些逻辑的技巧,达到了padding的效果。
大意是，如果取到的点是图像界外的，则默认是padding出来的像素。

熟练应用：
cpu下矩阵加速库：openblas和openblas
GPU下：cublas
在实际中，我们往往会采用指令集并行和GPU对矩阵乘法进行加速。
用指令集优化矩阵乘法难度比较大，非专业人士不建议采用，而且幸运的是我们现在有很多开源矩阵运算库，性能大概率超过手写的矩阵乘法。
CPU下的矩阵运算库有openblas和mkl等，GPU下有cublas，它们都是对fortran blas接口的实现。
在上述加速库的基础上，我们可以再对矩阵进行分割实现多核或者多机的并行。
在目前情况下，要实现快速的矩阵乘法，最便捷快速的方法就是用好openblas和cublas。

https://blog.csdn.net/ZhikangFu/article/details/78258393
cblas_sgemm(order,transA,transB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);

函数作用：C=alpha*A*B+beta*C

alpha =1,beta =0 的情况下，等于两个矩阵相成。

第一参数 oreder 候选值 有ClasRowMajow 和ClasColMajow 这两个参数决定一维数组怎样存储在内存中,

一般用ClasRowMajow。
参数 transA和transB ：表示矩阵A，B是否进行转置。候选参数 CblasTrans 和CblasNoTrans.

参数M：表示 A或C的行数。如果A转置，则表示转置后的行数

参数N：表示 B或C的列数。如果B转置，则表示转置后的列数。

参数K：表示 A的列数或B的行数（A的列数=B的行数）。如果A转置，则表示转置后的列数。

参数LDA：表示A的列数，与转置与否无关。 与在一维数组中的offset类似。

参数LDB：表示B的列数，与转置与否无关。（从caffe中看应该是与转置有关才对）

参数LDC：始终=N