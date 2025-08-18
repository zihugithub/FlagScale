# Megatron-LM 配置解析

**整体架构与并行策略**
- `tensor_model_parallel_size` 用于分布式深度学习框架中张量并行度的配置(简写为tp)
  - 用于将单个权重矩阵切分到多个GPU上存储和计算
  - 突破单卡显存限制，支持超大规模模型的训练
  - 实现数据并行和计算负载均衡，从而提升吞吐量并降低单卡内存压力
- `pipeline_model_parallel_size` 用于分布式训练/推理框架中流水线并行配置(简写为pp)
  - 利用所有设备的空闲时间，提高批量吞吐量
  - 把整个神经网络划分成连续的阶段, 不同阶段运行在不同设备组上运行
  - 前向传播时，第1阶段的输出作为第2阶段的输入
  - 常与张量并行结合使用，形成空间+时间的双维度加速
```
中小型集群：4张GPU部署70B参数模型时，  -pp=2与--tensor-parallel-size=2实现混合并行
大规模扩展：8张GPU处理175B参数模型时， -pp=4与--tensor-parallel-size=2实现混合并行
```
**优化器与稳定性控制**
- `disable_bias_linear` 用于控制神经网络中线性层是否包含偏置项
  - `True` 忽略所有线性层中的偏置项, 即不参与训练也不更新(形如 y = Wx + b 的操作，将强制使 b=0)
  - 大规模分布式训练中，偏置向量通常很小且对性能贡献有限，`去掉后可节省少量内存和计算资源`
- `use_distributed_optimizer` 用于控制分布式优化器的开启
  - 确保多机多卡环境下梯度同步正确无误，保证参数一致性，避免因不同设备间的梯度差异导致发散
  - 每次迭代时会跨所有工作者收集并平均梯度后再执行参数更新

**混合精度训练配置**
- `fp16` 用于控制是否开启半精度浮点数(FP16)进行前向/反向传播
  - 相比单精度(FP32)，显存占用减半、带宽需求降低，同时提升计算速度
  - 在低数值范围可能出现下溢问题
  - 使用16位二进制存储，包含1位符号位、5位指数位和10位尾数位，数值范围为[-65504, 65504]，分辨率约0.001
- `initial_loss_scale` 用于混合精度训练（尤其是FP16）中设置初始化缩放因子（设定初始的损失放大系数）
  - 通过在前向计算得到的Loss值乘以该系数，使得反向传播时的梯度同步放大，从而避免因FP16低精度导致的下溢问题
  - 复杂网络结构中，较大的初始缩放因子可抵消早期不稳定梯度的影响
  - `过度放大可能导致梯度爆炸型发散`
- `min_loss_scale` 用于混合精度训练（尤其是FP16）中，设定损失缩放系数的下限
  - 确保因发生梯度溢出导致动态调整缩小缩放因子时，也不会低于此阈值，从而维持数值稳定性
- `attention_softmax_in_fp32` 用于控制注意力机制中 Softmax 计算精度
  - `True` 强制将注意力分数（Attention Scores）的 Softmax 运算从默认的半精度模式（如 FP16）切换至单精度浮点数（FP32）进行
    - 中间结果会暂时以更高精度存储和处理，从而减少因数值精度不足导致的误差积累
    - FP32 的计算吞吐量低于 FP16，可能导致训练速度降低
    - 中间结果的类型提升会短暂增加内存压力，需配合梯度累加等策略缓解
```
自注意力机制涉及对指数级增长的值进行归一化操作，使用低精度格式（如 FP16）时，大梯度或极端激活值可能引发上溢/下溢问题。通过提升至 FP32，可以有效避免此类异常，确保概率分布的准确性和可解释性
```
- `accumulate_allreduce_grads_in_fp32` 用于决定是否在分布式训练中以FP32精度执行梯度的累加（Accumulation）和归约操作（AllReduce）
  - `True` 即使模型参数或中间计算使用低精度格式（如FP16/BF16），所有节点间的梯度同步过程仍会临时转换为FP32进行，从而避免因低精度导致的数值误差累积
    - FP32张量的传输会占用更多PCIe带宽，可能成为跨机架通信瓶颈

**日志记录与监控**
- `log_interval` 用于控制日志记录频率
  - 指定两次连续日志输出之间的时间间隔（单位为秒
- `no_log_loss_scale_to_tensorboard` 用于控制是否将自动调节的损失缩放因子上传至TensorBoard面板
  - 避免干扰主要监控曲线
**检查点保存策略**
- `no_save_optim` 不保存任何优化相关的状态, 仅保留模型权重参数
  - 能大幅减少存储开销（尤其对于大集群来说节省大量磁盘空间）
- `no_save_rng` 忽略随机数生成器的状态保存,
  - 下次加载模型时，DropPath等随机操作将重新开始，而非复现历史路径
  - 导致实验不可完全重现，但对预测任务影响较小；若追求严格复现能力应设为false
- `save_interval` 设置触发一次完整保存流程的时间间隔
- `tensorboard_log_interval` 指定向TensorBoard写入摘要数据的频率

```yaml
system:
  tensor_model_parallel_size: 2
  pipeline_model_parallel_size: 2
  disable_bias_linear: True
  use_distributed_optimizer: True
  precision:
    fp16: True
    initial_loss_scale: 522893
    min_loss_scale: 1.0
    attention_softmax_in_fp32: True
    accumulate_allreduce_grads_in_fp32: True
  logging:
    log_interval: 1
    no_log_loss_scale_to_tensorboard: true
  checkpoint:
    no_save_optim: true
    no_save_rng: true
    save_interval: 100000
    tensorboard_log_interval: 999999
```

# 模型参数配置解析
- `attention_backend` 深度学习框架中用于指定自注意力（Self-Attention）机制计算实现方式
  - 影响模型的训练速度、显存占用和推理效率
  - `unfused` 将传统注意力机制的各个步骤作为独立操作执行，不进行算子融合或内存访问优化
    - 常用于调试、基准测试或兼容性场景，因其逻辑清晰但效率较低
  - `FlashAttention` 通过分块计算（Tiling）、显存优化（减少临时存储）、并行化策略重构注意力内核，显著降低显存占用并提升吞吐量
```
attention_backend 决定使用哪种底层内核库或算法优化方案来计算自注意力（Self-Attention）
- 不同后端 对硬件特性（如GPU、CPU）、序列长度、批大小等场景进行了专门优化，从而平衡计算效率与资源利用率

技术本质：通过替换注意力计算函数，调用高度定制化的算子，减少内存拷贝、优化内存布局并充分利用并行计算能力
```
- `deterministic_mode` 深度学习框架中用于控制算法可重复性
  - `True` 强制所有潜在的非确定性操作切换到确定性算法
    - 无法找到对应的确定性版本，则会抛出运行时错误
- `use_mcore_models` 深度学习框架中用于启用基于MCore架构模型
  - `MCore模型` 通常针对特定硬件（如NVIDIA Hopper/Ampere架构GPU）进行了深度适配,以实现更高的计算吞吐量与显存利用率
    - 包含定制化算子融合、内存布局优化和通信压缩策略
- `transformer_impl` 指定Transformer模型的具体实现方式或变体
  - `transformer_engine` NVIDIA推出的一个高性能库，专门用于在GPU上加速Transformer模型的训练和推理, 显著降低显存占用并加速计算
  - `local` 采用本地默认的PyTorch原生实现，适用于通用场景但性能较低
- `num_layers` **控制模型深度**，设置模型中堆叠的相同子结构（如Transformer编码器层、LSTM单元等）的数量
  - 每一层会对输入数据进行非线性变换和特征提取
  - 随着层数的增加，模型能够逐步捕捉更复杂的模式
- `hidden_size` **控制模型宽度**，设置隐藏层的维度大小，即模型内部特征表示的向量长度
  - 决定了每个神经元在某一层级上能够存储和传递的信息量
  - 增加 hidden_size 可以提高模型对输入数据的理解深度
  - 随着 hidden_size 的增长，模型参数数量呈平方级增长，可能超出硬件承载范围，影响训练速度和可扩展性
- `num_attention_heads` 指定[多头注意力机制](../../../../../../../myself/多头注意力机制.md)中的头的数量
  - 若hidden_size=1024且num_attention_heads=16，则每个头的维度为1024/16=64
- `seq_length` 指定输入到模型中的单个样本的长度，即每个样本包含的token数量或时间步数
- `max_position_embeddings` 指定模型支持的最大位置编码长度，即输入序列的最大允许长度
  - 决定了位置嵌入矩阵的维度大小，通常以二维形式存在（形状为 (max_position_embeddings, hidden_size)），其中每一行对应一个特定位置的可学习向量
  - 超过长度的部分将被截断或分块处理
```
Transformer架构基于自注意力机制且不依赖递归结构，无法天然感知单词顺序
- 通过向输入添加位置编码，模型得以区分不同位置上的token，从而捕获语法和语义的逻辑关系
```
- `norm_epsilon` 归一化稳定项
  - 用于避免在归一化计算时分母为零的情况
  - 通过微小扰动改善反向传播过程中的数值稳定性，减少因极端值引发的梯度爆炸风险
- `use_rotary_position_embeddings` 用于控制模型是否采用基于旋转变换的位置编码
  - `True` 模型会在自注意力机制中引入相对位置信息，通过复数乘法形式的旋转操作来增强对序列顺序的感知能力, 能够更有效地捕捉token间的相对距离关系
- `no_position_embedding` 用于控制是否禁用位置编码
  - `True` 模型将不使用任何形式的位置编码（包括正弦/余弦绝对位置编码、可学习的位置嵌入等）
    - 输入序列中的词语仅保留其内容特征（如词义），而丢失了在原文中的顺序信息
    - 适用于无需考虑顺序的任务
  - 强制模型仅依赖语义而非位置进行决策
  - 减少可训练参数数量和计算开销，尤其在处理超长序列时能降低内存占用
- `swiglu` 用于控制是否使用激活函数（结合了Swish和GLU特点）【Swish激活函数和门控线性单元（GLU）】
  - `True` 增强非线性拟合能力，同时保持梯度流动顺畅
  - 通过Swish引入平滑且非单调的非线性变换，增强模型对复杂数据的拟合能力
- `multiple_of` 通道对齐约束
  - 确保各层宽度是256倍数，利于内存布局优化和缓存利用率最大化
- `normalization` 指定归一化方案
  - `RMSNorm` 通过计算输入向量的均方根进行归一化，而非传统的均值和标准差
    - RMS(x) = sqrt((1/D) * Σ_{i=1}^D x_i²) + ε
      - D是特征维度，ε是一个极小的平滑常数（如1e-6），用于防止除零错误
    - 归一化后的输出: y = x / (RMS(x) + ε)
    - 在视觉-语言融合模块中使用RMSNorm
  - `LayerNorm` 需计算均值和方差两次遍历数据，RMSNorm仅需一次平方求和操作
- `untie_embeddings_and_output_weights` 用于控制解绑嵌入矩阵与输出层权重
  - `权重绑定:` 模型共享输入词嵌入矩阵和输出层的线性投影权重的一种设计
  - 某些架构（如张量并行）要求各GPU卡上的输出维度必须严格匹配物理设备数量。此时，若保持权重绑定会导致跨设备的维度不一致错误。通过解耦参数，可将输出层的参数分布到多个计算节点上，解决并行度冲突问题
# 正则化与初始化策略
- `init_method_std` 指定初始化方法的标准差
  - `0.02` 属于保守策略，既避免过大初始值引发激活爆炸，又给予足够自由度供优化器探索
- `attention_dropout` 通常用于控制对注意力权重矩阵的随机屏蔽概率
  - 在计算得到的注意力权重矩阵上应用Dropout操作
  - 通过随机屏蔽部分注意力分数来防止过拟合，增强模型的泛化能力
  - `0.0` 在注意力机制中不应用Dropout操作,无论训练还是推理，模型始终使用完整的注意力权重
- `hidden_dropout` 用于控制神经网络中间层的随机失活概率
  - `0` 所有神经元均保留其输出值，不会在训练时被随机置零
    - 对应的Dropout层不会修改输入张量的任何元素，直接传递原始数据
- `weight_decay` 设置正则化的系数
  - 在损失函数中增加模型参数的平方和惩罚项,迫使权重趋向于更小的值，从而降低模型复杂度并防止过拟合
  - `0.1` 对权重的惩罚强度较高, 更强地限制权重增长，使模型更倾向于选择较小的参数值以最小化总损失
- `clip_grad`
- ``
- ``
- ``
- ``
- ``
- ``

```yaml
# model:
#   attention_backend: unfused
#   deterministic_mode: true
#   use_mcore_models: true
#   transformer_impl: transformer_engine
#   num_layers: 4
#   hidden_size: 512
  # num_attention_heads: 8
  # seq_length: 1024
  # max_position_embeddings: 1024
  # norm_epsilon: 1e-5
  # use_rotary_position_embeddings: true
  # no_position_embedding: true
  # swiglu: true
  # multiple_of: 256
  # normalization: RMSNorm
  # untie_embeddings_and_output_weights: true
  # init_method_std: 0.02
  # attention_dropout: 0.0
  # hidden_dropout: 0.0
  # weight_decay: 0.1
  # clip_grad: 1.0
  train_iters: 10
  micro_batch_size: 4
  global_batch_size: 1024
  seed: 42
  optimizer:
    weight_decay: 0.1
    adam_beta1: 0.9
    adam_beta2: 0.95
    lr_scheduler:
      lr: 2.0e-5
      min_lr: 2.0e-6
      lr_warmup_samples: 0
      lr_warmup_fraction: 0.01
      lr_decay_style: cosine

data:
  data_path: /home/gitlab-runner/data/pile_wikipedia_demo/pile_wikipedia_demo
  split: 1
  tokenizer:
    tokenizer_type: AquilaTokenizerFS
    vocab_file: ./examples/aquila/tokenizer/vocab.json
    merge_file: ./examples/aquila/tokenizer/merges.txt
    special_tokens_file: ./examples/aquila/tokenizer/special_tokens.txt
    vocab_size: 100008
```