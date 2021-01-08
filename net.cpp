//
// Created by wyl on 2020/11/26.
//

// https://blog.csdn.net/jinzhuojun/article/details/79834697
// https://www.cnblogs.com/yymn/articles/7498516.html
// https://blog.csdn.net/gaoenyang760525/article/details/72874816
// https://zhuanlan.zhihu.com/p/81667754
// https://blog.csdn.net/qq_28660035/article/details/80365570
// https://blog.csdn.net/ricky5000/article/details/68930978
// https://www.cnblogs.com/yymn/articles/6794796.html

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

template <typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter &param, NetParameter *param_filtered) {

}


/**AppendTop函数会向整个net的blob列表（blobs_）中添加一个新blob，同时将本层新建的top blob指向该新增blob，
 * 这样就把层的输出blob和blob列表(blobs_)关联起来了。AppendTop函数在新建blob时可能会采用同址计算（in-place computer），
 * 所谓同址计算就是同一层的top blob和bottom blob复用。
 * @param Dtype
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

/**
 * 1.给某层增加一个可学习参数blob(存放权重/偏置),放入params_, 同时放入learnable_params_;
 * 2.给某层增加一个params_lr_和params_weight_decay_,用来存放超训练参数
 * 3.一层中每个可学习参数blob(权重/偏置, learnable_params_)都对应有一个params_lr_和一个
 * params_weight_decay_, 超训练参数和可学习参数都是从LayerParameter中获取到.
 * 4.param_names_index_是对一层中可学习参数/超训练参数的总索引
 * @tparam Dtype
 * @param param
 * @param layer_id
 * @param param_id 第layer_id层第param_id个参数
 */
template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter &param, const int layer_id, const int param_id) {
    const LayerParameter& layer_param = layers_[layer_id]->layer_param();
    const int param_size = layer_param.param_size();
    // 如果param_name找不到，？？？？？
    string param_name = (param_size>param_id)?
            layer_param.param(param_id).name():"";
    //vector<string> param_display_names_，这里param_name获取的
    // 是PaParamSpec类型中的name成员，如果有name且非空,就把name压入该向量，否则就压入param_id
    if(param_name.size()){
        param_display_names_.push_back(param_name);
    }else{  // 如果字符串长度为0
        ostringstream param_display_name;
        param_display_name << param_name;
        param_display_names_.push_back(param_name.str());
    }
    const int net_param_id = params_.size();
    //layers_[layer_id]->blobs()中存放的是可学习参数(权重/偏置);
    // 一个层一般有两个blob,第一个存weight,第二个存bias
    //param_id_vecs_,存储的基本元素是net_param_id，每遍历一个参数blob,net_param_id和param_id_vecs_都会更新
    params_.push_back(layers_[layer_id]->blobs()[param_id]);
    param_id_vecs_[layer_id].push_back(net_param_id);   // 全局id
    param_layer_indices_.push_back(make_pair(layer_id, param_id));

    // 参数信息，里面包含学习率系数和参数衰减等信息(lr_mult, decay_mult)
    ParamSpec default_param_spec;
    const ParamSpec* param_spec = (layer_param.param_size()>param_id)?
            &layer_param.param(param_id):&default_param_spec;
    // 在caffe.proto的message ParamSpec里关于name的注释——>
    // To share a parameter between two layers, give it a (non-empty) name,
    // 可见，如果一个parameter是共享与多个网络层，那么它会有一个非空的name
    if (!param_size || !param_name.size() || (param_name.size() &&
    param_names_index_.find(param_name) == param_names_index_.end())){
        param_owners_.push_back(-1);
        // param_name非空，在param_names_index中找不到，则添加该param；
        // vector<int> param_owners_ 是一个存储parameter "onwer"的一个向量
        // ——> -1 表示当前Layer就是该parameter的"owner"
        if(param_name.size()){
            param_names_index_[param_name] = net_param_id;  // 存储共享参数，全局id
        }
        const int learnable_param_id = learnable_params_.size();
        learnable_params_.push_back(params_[net_param_id].get());
        learnable_param_ids_.push_back(learnable_param_id);
        has_params_lr_.push_back(param_spec->has_lr_mult());    // 是否有lr_mult
        has_params_decay_.push_back(param_spec->has_decay_mult());
        params_lr_.push_back(param_spec->lr_mult());
        params_weight_decay_.push_back(param_spec->decay_mult());
    }else{
        // 这个parameter已经存在于之前的某个或者某些网络层里，说明这个parameter是共享于多个layer
        const int owner_net_param_id = param_names_index_[param_name];  // 共享参数owner全局id
        param_owners_.push_back(owner_net_param_id);
        const pair<int, int>& owner_index =
                param_layer_indices_[owner_net_param_id];
        const int owner_layer_id = owner_index.first;       // owner所在layer_id
        const int owner_param_id = owner_index.second;      // owner在layer_id层的局部id
        LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
                                           << "' owned by "
                                           << "layer '" << layer_names_[owner_layer_id] << "', param "
                                           << "index " << owner_param_id;
        Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();
        Blob<Dtype>* owner_blob =
                layers_[owner_layer_id]->blobs()[owner_param_id].get();
        const int param_size = layer_param.param_size();

        if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                      ParamSpec_DimCheckMode_PERMISSIVE)) {
            // Permissive dimension checking -- only check counts are the same.
            CHECK_EQ(this_blob->count(), owner_blob->count())
                    << "Cannot share param '" << param_name << "' owned by layer '"
                    << layer_names_[owner_layer_id] << "' with layer '"
                    << layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
                    << "shape is " << owner_blob->shape_string() << "; sharing layer "
                    << "shape is " << this_blob->shape_string();
        } else {
            // Strict dimension checking -- all dims must be the same.
            CHECK(this_blob->shape() == owner_blob->shape())
                    << "Cannot share param '" << param_name << "' owned by layer '"
                    << layer_names_[owner_layer_id] << "' with layer '"
                    << layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
                    << "shape is " << owner_blob->shape_string() << "; sharing layer "
                    << "expects shape " << this_blob->shape_string();
        }

        //获取owner layer的learnable_param_id，并且压入当前layer的向量learnable_param_ids_。
        //而且在这里也没有把参数blob压入learnable_params_向量（只是将id压入learnable_param_ids_），
        // 从而避免当前layer与sharing layer之间关于shared parameter blob 的重复
        // 因为复用, 需要把learnable_param_id再重复放入learnable_param_ids_中
        const int learnable_param_id = learnable_param_ids_[owner_net_param_id];
        learnable_param_ids_.push_back(learnable_param_id);
        if(param_spec->has_lr_mult()){
            if(has_params_lr_[learnable_param_id]){
                CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
                    << "Shared param '" << param_name << "' has mismatched lr_mult.";
            }else{
                has_params_lr_[learnable_param_id] = true;
                params_lr_[learnable_param_id] = param_spec->lr_mult();
            }
        }
        if (param_spec->has_decay_mult()) {
            if (has_params_decay_[learnable_param_id]) {
                CHECK_EQ(param_spec->decay_mult(),
                         params_weight_decay_[learnable_param_id])
                    << "Shared param '" << param_name << "' has mismatched decay_mult.";
            } else {
                has_params_decay_[learnable_param_id] = true;
                params_weight_decay_[learnable_param_id] = param_spec->decay_mult();
            }
        }
    }
}

template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int layer_id) {
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
        const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
        const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
        const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
        LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", top blob " << blob_name
        << " data: " << data_abs_val_mean;
    }
    for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
         ++param_id) {
        const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
        const int net_param_id = param_id_vecs_[layer_id][param_id];
        const string& blob_name = param_display_names_[net_param_id];
        const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
        LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << blob_name
        << " data: " << data_abs_val_mean;
    }
}

// 执行从start层到end层的前向传递，采用简单的for循环调用。
// 在前向传播之前为什么要before_forward_和back_forward_????
template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
    CHECK_GE(start, 0);
    CHECK_GT(end, layers_.size());
    Dtype loss = 0;
    for(int i=0; i<=end; i++){
        for(int c=0; c<before_forward_.size(); c++){
            before_forward_[c]->run(i);
        }
        Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
        loss += layer_loss;
        if(debug_info_) {ForwardDebugInfo(i);}
        for(int c=0; c<after_backward_.size(); c++){
            after_backward_[c]->run(i);
        }
    }
    return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
    return ForwardFromTo(0, layers_.size()-1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
    return ForwardFromTo(0, layers_.size()-1);
}

/*
 * bottom_vecs_前向计算得到top_vecs_, 并计算loss
 */
template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype *loss) {
    if(loss != NULL){
        *loss = ForwardFromTo(0, layers_.size()-1);
    }else{
        ForwardFromTo(0, layers_.size()-1);
    }
    return net_output_blobs_;
}

// !!!!!!!!!!!!!!!
/*
功能：把网络输入层的blob读到net_input_blobs_，然后前向计算loss
*/
template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(const vector<Blob<Dtype> *> &bottom, Dtype *loss) {
    LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: Forward(bottom, loss) "
                               << "will be removed in a future version. Use Forward(loss).";
    for(int i=0; i<bottom.size(); i++){
        net_input_blobs_[i]->CopyFrom(*bottom[i]);  // 将bottom数据复制到net_input_blobs中 cudaMemcpy
    }
    return Forward(loss);
}

template <typename Dtype>
void Net<Dtype>::BackwardDebugInfo(const int layer_id) {
    const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
    for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
        if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
        const Blob<Dtype>& blob = *bottom_vec[bottom_id];
        const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
        LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", bottom blob " << blob_name
        << " diff: " << diff_abs_val_mean;
    }
    for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
         ++param_id) {
        if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
        const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
        const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
        LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << param_id
        << " diff: " << diff_abs_val_mean;
    }
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo(int start, int end) {
    CHECK_GE(end, 0);
    CHECK_LT(end, layers_.size());
    for(int i=start; i>=end; --i){
        for(int c=0; c<before_backward_.size(); c++){
            before_backward_[c]->run(i);
        }
        if(layer_need_backward_[i]){
            layers_[i]->Backward(top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);
            if (debug_info_) {BackwardDebugInfo(i); }
        }
        for(int c=0; c<after_backward_.size(); c++){
            after_backward_[c]->run(i);
        }
    }
}

template <typename Dtype>
void Net<Dtype>::BackwardFrom(int start) {
    BackwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::BackwardTo(int end) {
    BackwardFromTo(layers_.size() - 1, end);
}

template <typename Dtype>
void Net<Dtype>::Backward() {
    BackwardFromTo(layers_.size()-1, 0);
    if (debug_info_) {
        Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
        for (int i = 0; i < learnable_params_.size(); ++i) {
            asum_data += learnable_params_[i]->asum_data();
            asum_diff += learnable_params_[i]->asum_diff();
            sumsq_data += learnable_params_[i]->sumsq_data();
            sumsq_diff += learnable_params_[i]->sumsq_diff();
        }
        const Dtype l2norm_data = std::sqrt(sumsq_data);
        const Dtype l2norm_diff = std::sqrt(sumsq_diff);
        LOG(ERROR) << "    [Backward] All net params (data, diff): "
                   << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
                   << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
    }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
    for(int i=0; i<layers_.size(); i++){
        layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
    }
}


};