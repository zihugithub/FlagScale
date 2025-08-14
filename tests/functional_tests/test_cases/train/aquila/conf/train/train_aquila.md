# **Megatron-LM 配置解析**

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


```yaml
system:
  # tensor_model_parallel_size: 2
  # pipeline_model_parallel_size: 2
  # disable_bias_linear: True
  # use_distributed_optimizer: True
  # precision:
  #   fp16: True
  #   initial_loss_scale: 522893
  #   min_loss_scale: 1.0
  #   attention_softmax_in_fp32: True
  #   accumulate_allreduce_grads_in_fp32: True
  logging:
    log_interval: 1
    no_log_loss_scale_to_tensorboard: true
  checkpoint:
    no_save_optim: true
    no_save_rng: true
    save_interval: 100000
    tensorboard_log_interval: 999999
```

```yaml
model:
  attention_backend: unfused
  deterministic_mode: true
  use_mcore_models: true
  transformer_impl: transformer_engine
  num_layers: 4
  hidden_size: 512
  num_attention_heads: 8
  seq_length: 1024
  max_position_embeddings: 1024
  norm_epsilon: 1e-5
  use_rotary_position_embeddings: true
  no_position_embedding: true
  swiglu: true
  multiple_of: 256
  normalization: RMSNorm
  untie_embeddings_and_output_weights: true
  init_method_std: 0.02
  attention_dropout: 0.0
  hidden_dropout: 0.0
  weight_decay: 0.1
  clip_grad: 1.0
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