//
// Created by wyl on 2020/11/26.
//


// https://www.cnblogs.com/yymn/articles/7498516.html
// https://blog.csdn.net/gaoenyang760525/article/details/72874816
// https://zhuanlan.zhihu.com/p/81667754
// https://blog.csdn.net/qq_28660035/article/details/80365570
// https://blog.csdn.net/ricky5000/article/details/68930978
#include "include/net.hpp"
namespace caffe{
template<typename Dtype>
Net<Dtype>::Net(const NetParameter &param) {
    Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string &param_file, Phase phase) {

}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter &in_param) {
    phase_ = in_param.state().phase();
    NetParameter filtered_param;
    FilterNet(in_param, &filtered_param);

    NetParameter param;
    InsertSplits(filtered_param, &param);

    // 建立所有层
    name_ = param.name();
    map<string, int> blob_name_to_idx;
    set<string> available_blobs;
    memory_used_ = 0;

    //设置每层的输入输出
    bottom_vecs_.resize(param.layer_size());
    top_vecs_.resize(param.layer_size());
    bottom_id_vecs_.resize(param.layer_size());
    top_id_vecs_.resize(param.layer_size());
    param_id_vecs_.resize(param.layer_size());  // 权重blob的id
    bottom_need_backward_.resize(param.layer_size());   // blob是否需要反向传播

    for(int layer_id=0; layer_id<param.layer_size(); layer_id++){
        // 如果没有设置层的phase，就从网络的phase继承，因为能通过FilterNet留下来肯定符合该phase
        if(!param.layer(layer_id).has_phase()){
            param.mutable_layer(layer_id)->set_phase(phase_);
        }

        // 设置层参数
        const LayerParameter& layer_param = param.layer(layer_id);
        if(layer_param.propagate_down_size()>0){
            CHECK_EQ(layer_param.propagate_down_size(),
                     layer_param.bottom_size())
                    << "propagate_down param must be specified "
                    << "either 0 or bottom_size times ";
        }
        // LayerRegistry?????????
        /*
        *把当前层的参数转换为shared_ptr<Layer<Dtype>>，
        *创建一个具体的层，并压入到layers_中
        */
        layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
        layer_names_.push_back(layer_param.name());
        LOG_IF(INFO, Caffe::root_solver())
                << "Creating Layer " << layer_param.name();
        bool need_backward = false;

        // bottom_size()？？？？
        for(int bottom_id = 0; bottom_id<layer_param.bottom_size(); ++bottom_id){
            const int blob_id = AppendBottom(param, layer_id, bottom_id,
                    &available_blobs, &blob_name_to_idx);
            need_backward |= blob_need_backward_[blob_id];
        }

        int num_top = layer_param.top_size();
        for(int top_id = 0; top_id<num_top; ++top_id){
            //通过AppendTop和AppendBottom, bottom_vecs_和top_vecs_连接在了一起
            //在AppendTop中会往available_blobs添加某层的输出blob,在AppendBottom中会
            //从available_blobs中删除前一层的输出blob，所有layers遍历完后剩下的就
            //是整个net的输出blob
            AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
            if(layer_param.type() == "Input"){
                const int blob_id = blobs_.size() - 1;
                net_input_blob_indices_.push_back(blob_id);
                net_input_blobs_.push_back(blobs_[blob_id].get());
            }
        }
        // 如果设置了自动生成top blob,具体可参看关于之前关于layer的博文:
        // 自动创建top blobs来满足ExactNumTopBlobs()和MinTopBlobs()的需要
        Layer<Dtype>* layer = layers_[layer_id].get();
        if(layer->AutoTopBlobs()){
            const int need_num_top =
                    std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
            for(; num_top<need_num_top; num_top++){
                AppendTop(param, layer_id, num_top, NULL, NULL);
            }
        }

        // 前面创建了具体的层，并为层创建了输入bottom blob 和输出top blob。
        // 层都连接起来后，就调用层的SetUp函数，输入bottom blob 和top blob 的智能指针，建立层。
        //setup()函数的功能是为创建的参数blob分配数据内存空间，如有必要还需要调整该层的输入bottom blob 和输出top blob的shape。
        layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
        LOG_IF(INFO, Caffe::root_solver())
                << "Setting up " << layer_names_[layer_id];
        // 遍历第layer_id层的top blob
        // blob_loss_weights_？？？？
        // 有多少个top_id_vecs_就需要多少blob_loss_weights_
        for(int top_id = 0; top_id<top_vecs_[layer_id].size(); ++top_id ){
            if(blob_loss_weights_.size() < top_id_vecs_[layer_id].size()){
                blob_loss_weights_.resize(top_id_vecs_[layer_id].size()+1, Dtype(0));
            }
            blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
            LOG_IF(INFO, Caffe::root_solver())
                    << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
            if(layer->loss(top_id)){
                LOG_IF(INFO, Caffe::root_solver())
                        << "    with loss weight " << layer->loss(top_id);
            }
            // 调用blob类的count函数来，来计算占用的空间。
            memory_used_ += top_id_vecs_[layer_id][top_id]->count();
        }
        // log输出中常见的Memory required for data:
        LOG_IF(INFO, Caffe::root_solver())
                << "Memory required for data: " << memory_used_ * sizeof(Dtype);

        const int param_size = layer_param.param_size();
        // 第layer_id层blob的个数
        const int num_param_blobs = layers_[layer_id]->blobs().size();
        //param_size是Layermeter类型对象layer_param中ParamSpec param成员的个数,
        //num_param_blobs是一个Layer中learnable parameter blob的个数，
        // 要 param_size <= num_param_blobs
        CHECK_LE(param_size, num_param_blobs)
                << "Too many params specified for layer " << layer_param.name();
        ParamSpec default_param_spec;
        for(int param_id = 0; param_id<num_param_blobs; param_id++){
            const ParamSpec* param_spec = (param_id<param_id)?
                    &layer_param.param(param_id):&default_param_spec;
            //学习率不为0则为需要反向传播
            const bool param_need_backward = param_spec->lr_mult()!=0;
            need_backward |= param_need_backward;
            layers_[layer_id]->set_param_propagate_down(param_id,
                    param_need_backward);
        }

        for(int param_id = 0; param_id< num_param_blobs; ++param_id){
            // 为网络增加新的参数blob，只加有参数的层的param blob
            // 对于某些有参数的层，例如：卷基层、全连接层有weight和bias。
            // 该函数主要是修改和参数有关的变量，实际的层参数的blob在上面提到的setup()函数中已经创建。如：将层参数blob的指针压入到params_。
            AppendParam(param, layer_id, param_id);
        }
        layer_need_backward_.push_back(need_backward);
        if(need_backward) {
            for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); top_id++) {
                blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
            }
        }
    }
    /*至此上面部分各个层被创建并启动，下面部分是按后向顺序修正backward设置  */
    // computation for the entire layer
    // 之前都是前向依次设置反向的，下面的是按后向顺序修正前向设置：
    // 可以跳过对loss没贡献层的反向计算，同时检查是否所有bottom blob都需要反向计算。
    // 因此，定义了两个set来存需要/不需要反向的blob的名字
    set<string> blobs_under_loss;
    set<string> blobs_skip_backcp;

    for(int layer_id = layers_.size(); layer_id>=0; layer_id--){
        bool layer_contributes_loss = false;
        bool layer_skip_propagate_down = true;
        // 输出blobs是否对loss有贡献 来判断layer是否需要反传
        for(int top_id=0; top_id < top_vecs_[layer_id].size(); top_id++){
            // blob_names_整个网络中，所有非参数blob的name
            const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
            if(layers_[layer_id]->loss(top_id) || (blobs_under_loss.find(blob_name)!=blobs_under_loss.end())){
                layer_contributes_loss = true;
            }
            if(blobs_skip_backcp.find(blob_name) == blobs_skip_backcp.end()){
                layer_skip_propagate_down = false;
            }
            // 该层对loss有贡献且该层不跳过反向传播
            if(layer_contributes_loss && !layer_skip_propagate_down)
                break;
        }

        // 以layer_skip_propagate_down为基准, 修改bottom_need_backward_
        if(layer_need_backward_[layer_id] && layer_skip_propagate_down){
            layer_need_backward_[layer_id] = false;
            for(int bottom_id = 0; bottom_id<bottom_vecs_[layer_id].size(); ++bottom_id){
                bottom_need_backward_[layer_id][bottom_id] = false;
            }
        }

        if(!layer_contributes_loss) {layer_need_backward_[layer_id] = false;}

        if (Caffe::root_solver()) {
            if (layer_need_backward_[layer_id]) {
                LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
            } else {
                LOG(INFO) << layer_names_[layer_id]
                          << " does not need backward computation.";
            }
        }
        for(int bottom_id=0; bottom_id<bottom_vecs_[layer_id].size(); bottom_id++){
            if(layer_contributes_loss){
                const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
                blobs_under_loss.insert(blob_name);
                //判断当前层是否contributions to loss 是的话 就把名字插入 blobs_under_loss中
            }else{
                bottom_need_backward_[layer_id][bottom_id] = false;
            }
            if(!bottom_need_backward_[layer_id][bottom_id]){
                const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
                blobs_skip_backcp.insert(blob_name);
                // 若本层不需要反向传播，将名字插入blobs_skip_backp中
            }
        }
    }

    if(param.force_backward()){
        for(int layer_id = 0; layer_id<layers_.size(); layer_id++){
            layer_need_backward_[layer_id] = true;
            for(int bottom_id = 0; bottom_id<bottom_need_backward_[layer_id].size(); ++bottom_id){
                bottom_need_backward_[layer_id][bottom_id] = bottom_need_backward_[layer_id][bottom_id] ||
                        layers_[layer_id]->AllowForceBackward(bottom_id);
                blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
                        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
                        bottom_need_backward_[layer_id][bottom_id];
            }
            for(int param_id=0; param_id<layers_[layer_id]->blobs().size();
            ++param_id){
                layers_[layer_id]->set_param_propagate_down(param_id, true);
            }
        }
    }

    // In the end, all remaining blobs are considered output blobs.
    // 最终，所有还在available_blobs中的blob都会被视为输出
    for(set<string>::iterator it=available_blobs.begin();
    it!=available_blobs.end(); it++){
        LOG_IF(INFO, Caffe::root_solver())
                << "This network produces output " << *it;
        net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
        net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
    }

    for(size_t blob_id = 0; blob_id<blob_names_.size(); ++blob_id){
        blob_names_index_[blob_names_[blob_id]] = blob_id;
    }

    for(size_t layer_id = 0; layer_id<layer_names_.size(); layer_id++){
        layer_names_index_[layer_names_[layer_id]] = layer_id;
    }

    SharedWeights();
    debug_info_ = param.debug_info();
    LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
}

/**AppendTop函数会向整个net的blob列表（blobs_）中添加一个新blob，同时将本层新建的top blob指向该新增blob，
 * 这样就把层的输出blob和blob列表(blobs_)关联起来了。AppendTop函数在新建blob时可能会采用同址计算（in-place computer），
 * 所谓同址计算就是同一层的top blob和bottom blob复用。
 * @tparam Dtype
 * @param param
 * @param layer_id
 * @param bottom_id
 * @param available_blobs
 * @param blob_name_to_idx
 */
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter &param, const int layer_id, const int top_id,
                           set <string> *available_blobs, map<string, int> *blob_name_to_idx) {
    // ????????这种写法？？
    // 需要熟悉protobuf的写法
    shared_ptr<LayerParameter> layer_param(new LayerParameter(param.layer(layer_id)));
    const string& blob_name = (layer_param->top_size() > top_id) ?
            layer_param->top(top_id) : "(automatic)";


    //同址计算:top blob使用和bottom blob相同的地址和id
    //是否使用同址计算由prototxt中对top/bottom blob名字的定义决定
    if(blob_name_to_idx && layer_param->bottom_size()>top_id &&
    layer_param->bottom(top_id) == blob_name){
        // In-place computation
        LOG_IF(INFO, Caffe::root_solver())
        << layer_param->name() << " -> " << blob_name << " (in-place)";
        top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
        top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
    }else if(blob_name_to_idx && blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()){
        LOG(FATAL) << "Top blob '" << blob_name
                   << "' produced by multiple sources.";
    }else{
        // 不需要同址计算，则要创建一个新blob
        shared_ptr<Blob<Dtype>> blob_pointer(new Blob<Dtype>);
        const int blob_id = blobs_.size();
        blobs_.push_back(blob_pointer);
        blob_names_.push_back(blob_name);
        blob_need_backward_.push_back(false);
        if(blob_name_to_idx){
            (*blob_name_to_idx)[blob_name] = blob_id;
        }
        top_vecs_[layer_id].push_back(blob_pointer.get());
        top_id_vecs_[layer_id].push_back(blob_id);
    }
    if(available_blobs){
        available_blobs->insert(blob_name);
    }
}


/**　往第layer_id层的bottom_id位置设置输入向量，则availabel_blobs删掉该向量；设置该向量是否反传(默认false);返回该向量在blobs_中的位置
 * AppendBottom函数不会向blobs_新增blob了，只是简单的把新增的bottom blob和在AppendTop中已经增加的blobs_关联起来。
经过上述两个函数的处理，前一层的top blob、当前层的bottom blob就通过blobs_关联起来了，整个net中所有的层级就连结到一起。
 * @tparam Dtype
 * @param param
 * @param layer_id
 * @param bottom_id
 * @param available_blobs
 * @param blob_name_to_idx
 * @return 返回blob_id
 */
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter &param, const int layer_id, const int bottom_id,
                             set <string> *available_blobs, map<string, int> *blob_name_to_idx) {
    const LayerParameter& layer_param = param.layer(layer_id);
    const string& blob_name = layer_param.bottom(bottom_id);
    if (available_blobs->find(blob_name) == available_blobs->end()) {
        LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
                   << layer_param.name() << "', bottom index " << bottom_id << ")";
    }
    const int blob_id = (*blob_name_to_idx)[blob_name];

    LOG_IF(INFO, Caffe::root_solver())
    << layer_names_[layer_id] << " <- " << blob_name;

    //新增一个blob的动作是在top中完成的, bottom中只是把当前层bottom和前一层
    //top的地址连接起来(通过bottom/top指向相同的blobs_[id]/blob_id来连接)
    top_vecs_[layer_id].push_back(blobs_[blob_id].get());
    top_id_vecs_[layer_id].push_back(blob_id);
    if(available_blobs){
        available_blobs->erase(blob_name);  //
    }

    bool need_backward = blob_need_backward_[blob_id];
        // Check if the backpropagation on bottom_id should be skipped
    if(layer_param.propagate_down_size()>0){
        need_backward = layer_param.propagate_down(bottom_id);
    }
    bottom_need_backward_[layer_id].push_back(need_backward);
    return blob_id;
}

template <typename Dtype>
void Net<Dtype>::AppenParam(const NetParameter &param, const int layer_id, const int param_id) {
    
}

};