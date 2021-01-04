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
  const string& blob_name = layer_param.bottom(bottom_id);　// 通过bottom获取blob_name
```

**AppenBottom**

```
往第layer_id层的bottom_id位置设置输入向量，则availabel_blobs删掉该向量；设置该向量是否反传(默认false);返回该向量在blobs_中的位置
```
**AppenParam**
params_ 存储整个Net的参数，其中包括各Layer中的blobs_
layer_id, param_id
param_id_vecs_ 存储各层param的id
param_layer_indices_ 存储