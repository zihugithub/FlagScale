**defaults(默认配置组)**

[Hydra](../../../../../../myself/hydra.md) 会合并多个配置文件的内容：

- `_self_` 确保文件本身的配置被纳入最终结果
- `inference: tp2` 加载外部资源（如 conf/inference/tp2.yaml）,覆盖或补充当前配置
```yaml
defaults:
  - _self_
  - inference: tp2
```
**experiment (实验详情)**

- `exp_name` 当前实验的名称
- `exp_dir` 实验结果存储目录
- `task` 任务类型及实现方式
  - `type` 明确任务模式
  - `backend` 指定后端框架(如：使用 vllm 作为推理引擎)
  - `entrypoint` 指定主程序入口文件
- `runner` 运行时环境（用于定义实验执行时的运行环境和资源管理策略）
  - `hostfile` 不依赖 MPI/Hostfile 进行进程间通信（由 PyTorch NCCL 直接处理）
    - `null` 未指定主机文件（hostfile），即不使用分布式计算中的多节点配置
    - 单卡或单机训练/推理时，通常不需要设置此参数；若涉及多机集群部署，则需提供包含各节点IP地址等信息的文件来实现跨机器通信
    - 设为 null 表明当前任务仅在一个计算节点上运行
- `cmds` 预执行的命令序列
  - `before_start` 主程序启动前运行的命令序列
- `envs` 系统级环境变量注入
  - `HYDRA_FULL_ERROR` Hydra严格模式：任何未识别的配置项立即报错退出
  - `CUDA_VISIBLE_DEVICES` 指定启用的GPU显卡
  - `CUDA_DEVICE_MAX_CONNECTIONS` 指定每个设备（GPU）的物理连接数
    - 设为 1 可规避因过多上下文切换导致的内存碎片化问题，适合高频次小规模推理任务
  - `CUBLAS_WORKSPACE_CONFIG: ":4096:8"` CuBLAS库的工作内存池大小配置（提升矩阵运算效率）
    - `:4096:8` 采用三元组格式 <size>:<count> , 表示为每个线程块预分配 4096KB 的工作缓存区，并允许最多 8个并行请求
  - `NCCL_ALGO` 指定多机通信采用的算法
    - `Ring` 环形拓扑算法，所有参与节点形成一个闭合环路，数据沿固定方向顺序传递
      - 仅与直接相邻的两个节点通信, 形成单向循环链路
      - 随着节点数增加，通信路径长度线性增长，可能导致整体延迟上升
      - 适用于中小等规模集群，简单稳定，便于调试和验证算法正确性
    - `Tree` 树形拓扑算法，通过父子节点逐级聚合或分发数据
      - 适用于大规模分布式训练，可减少跳数并利用分层结构降低延迟
    - `Collnet` CollNet专用算法, 针对特定硬件优化的网络结构，可能结合了混合并行策略（需硬件支持）
    - `NVLS` NVLS低延迟模式, 用于对实时性要求极高的场景（如高频交易中的梯度更新）
      - 适用于实时交互式应用，优先保证最低延迟，接受一定程度的带宽损耗换取响应速度
  - `NVTE_APPLY_QK_LAYER_SCALING` 控制是否在[注意力机制](../../../../../../myself/NVTE.md)中对查询（Query, Q）和键（Key, K）进行分层缩放
    - 在标准的注意力机制中，Q和K的点积结果可能因维度过高导致数值不稳定（如梯度消失/爆炸）
    - 分层缩放则允许为每个网络层动态分配不同的缩放比例
    - `0` 禁用分层缩放, 默认使用全局缩放策略, 适用于小型模型、简单任务
    - `1` 启用分层缩放, 按层动态调整Q/K的缩放因子， 适用于超深网络（>100层）、长序列建模
  - `NVTE_ALLOW_NONDETERMINISTIC_ALGO` 允许非确定性算法, 即系统在运行时可能采用带有随机性的计算路径以换取性能提升
    - `0` 适用于生产环境部署，牺牲部分理论峰值性能 来 确保跨设备一致性
    - `1` 快速原型验证场景，迭代速度提升20%~50%*，但结果不可复现
    - `1` 超大规模训练场景，缓解通信瓶颈导致的闲置等待，需配合校验点快照机制
  - `NVTE_FLASH_ATTN` 控制是否启用基于 FlashAttention 的注意力机制优化
    - `0` 沿用标准实现
      - 适用于调试验证场景，确保中间结果可观测性，便于排查异常
    - `1` 采用 FlashAttention 替代传统的自注意力计算方式，以提升计算效率和内存利用率
      - 适用于长文本生成，支持更长的上下文窗口（如4K+ tokens），适合LLM推理任务
      - 适用于大规模训练，批量增大时显存需求更低，可提升吞吐量
    - FlashAttention 是一种针对长序列设计的高效注意力算法
      - **分块计算** 将大尺寸的 Query/Key/Value 矩阵分割成小块，逐次加载到 GPU 的高速缓存（SRAM）中进行局部计算，减少 HBM（高带宽内存）访问次数
      - **重计算** 在反向传播阶段不存储中间结果，而是动态重新计算所需梯度，进一步节省内存开销
      - **精确性保证** 尽管优化了 I/O 路径，但输出结果与标准注意力机制完全等价，避免因近似带来的精度损失
  - `NVTE_FUSED_ATTN` 控制是否启用 融合注意力 的操作，将多个操作步骤合并为单一内核函数以减少计算开销
    - `0` 适用于实验研究，保持标准PyTorch接口兼容性，性能损失但可解释性强
    - `1` 适用于高吞吐推理场景，端到端延迟降低30%~50%*，调试困难（难以插入断点观察中间结果）
    - `1` 适用于大规模训练场景，显存占用减少20%~40%，需验证数值稳定性
  - `CUDNN_BENCHMARK` 用于控制是否启用 cuDNN 库的自动算法搜索功能
    - 在训练或推理前对不同卷积实现进行性能测试，选择当前硬件配置下的最优算法路径
    - 等效地通过 PyTorch 接口 torch.backends.cudnn.benchmark
    - **首次运行开销增加** 设置为 True 时，程序会在初次执行卷积操作前遍历所有候选算法
      - 记录各方案的实际耗时并锁定最快者，会引入额外时间成本，但后续迭代中因缓存了最优解而大幅提升效率
    - **输入稳定性依赖性** 当网络结构的输入维度（如 batch size、图像尺寸）完全固定时有效
      - 输入动态变化，每次形状变更都会触发重新基准测试，反而导致性能下降
    - **与确定性的互斥性** 该模式与 cudnn.deterministic=True 冲突
      - benchmark 允许非确定性的算法选择策略以追求速度最大化
    - `True` 适用于静态输入任务，固定形状的数据增强、图像分类等场景下可提升吞吐量，需要确保输入尺寸在整个生命周期不变
    - `False` 适用于研究实验，便于调试中间结果，避免因算法切换导致的数值波动，需要牺牲部分性能换取可控性
    - `False` 适用于动态网络架构，RNN/Transformer 等变长序列模型必须关闭此选项，频繁的配置开销会抵消加速收益
  - `CUDNN_DETERMINISTIC` 用于控制是否强制 cuDNN 使用确定性算法
    - **随机种子固定** cuDNN 内部涉及并行计算优化时可能引入非确定性（如线程调度顺序、浮点运算精度差异）
      - 启用该选项后，库会锁定随机数生成器的初始状态，确保每次运行的计算路径完全一致
    - **算法选择限制** 某些高性能但非确定性的实现会被排除在外，转而使用稳定但较慢的替代方案
      - 导致微小的性能损耗，但换来了结果的严格一致性
    - **与基准测试模式互斥** 若同时开启 cudnn.benchmark=True（自动搜索最优算法），可能因每次选取不同算法破坏可复现性,
      - 因此二者通常不共存
    - `True` 所有基于 cuDNN 的操作（如卷积、池化等）将产生可重复的结果，即相同输入和参数下输出始终一致
      - 适用于 研究论文复现 确保实验结果严格一致，便于同行评审验证，可能导致轻微性能下降
      - 适用于 模型调试 定位数值误差来源更直观，排除随机因素干扰，应避免依赖特定硬件架构特性
    - `False` 适用于 生产部署 最大化吞吐量，允许动态调整算法以适应多变的工作负载，需配合版本控制保证环境稳定
 

```yaml
experiment:
  exp_name: deepseek
  exp_dir: tests/functional_tests/test_cases/inference/deepseek/results_test/tp2
  task:
    type: inference
    backend: vllm
    entrypoint: flagscale/inference/inference_aquila.py
  runner:
    hostfile: null
  cmds:
    before_start:
      source /root/miniconda3/bin/activate flagscale-inference
  envs:
    HYDRA_FULL_ERROR: 1
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
    CUDA_DEVICE_MAX_CONNECTIONS: 1
    CUBLAS_WORKSPACE_CONFIG: ":4096:8"
    NCCL_ALGO: "Ring"
    NVTE_APPLY_QK_LAYER_SCALING: 0
    NVTE_ALLOW_NONDETERMINISTIC_ALGO: 0
    NVTE_FLASH_ATTN: 0
    NVTE_FUSED_ATTN: 0
    CUDNN_BENCHMARK: "false"
    CUDNN_DETERMINISTIC: "true"
```

**action (执行动作)**

Hydra 配置文件中，action: run 是最核心的指令之一，它明确告诉系统执行什么操作，会触发以下一系列自动化过程：
- 根据 defaults 加载基础设置；
- 解析 experiment 部分定义的任务细节（如模型、数据、环境等）；
- 按照 cmds 中的前置命令准备运行环境；
- 应用 envs 中的环境变量配置；
- 最终调用 entrypoint 指向的脚本文件开始推理过程

**触发入口点 (Entry Point)** 当 action=run 被激活时，Hydra 会根据 entrypoint 指定的主程序入口文件，将其视为主要逻辑载体

**依赖注入与初始化顺序** 跳过这一步，可能导致模块缺失或函数行为异常，尤其是当不同项目共享同一台服务器
- runner.hostfile: null 表明这是一个单机单节点的任务，不需要分布式集群支持。因此，Hydra 不会尝试跨机器协调资源，而是直接在本机当前环境中启动进程
- cmds.before_start 里的 source /root/miniconda3/bin/activate flagscale-inference 确保在运行主程序前先激活特定的 Conda 虚拟环境，保证了后续所有操作都在正确的包版本下进行

**环境变量生效时机** 在运行时被自动注入envs指定的参数变量到进程中的环境里，控制硬件行为模式

**输出管理规范** hydra.run.dir: ${experiment.exp_dir}/hydra 定义了日志和缓存文件的存储位置。每次运行都会创建一个唯一子目录，里面保存着完整的执行记录（包括超参数、中间结果、错误堆栈跟踪等信息）

```yaml
action: run
```

**hydra (框架专属控制)**

- 路径解析机制：利用模板语法 ${...} 实现变量插值。此处会把前面定义的 experiment.exp_dir（即那个 tp4 路径）扩展出来，形成类似路径：tests/functional_tests/test_cases/inference/deepseek/results_test/tp4/hydra。所有的临时文件、元数据都将写入此处。
- 用途：方便后续追踪不同实验之间的差异，同时保持项目结构清晰
- 理解 Hydra 的配置分层机制, Hydra 将配置分为两类：
  - 用户定义的配置（如 experiment, action 等）：是 YAML 文件中显式声明的部分
  - 内部元数据配置（以 hydra: 开头）：用于控制 Hydra 自身行为（如输出目录、日志级别等），**默认不会合并到主配置对象中**
  - 通过 def main(config: DictConfig) 获取的是用户空间的配置树，而 hydra: 相关的设置属于框架内部的“运行时选项”，存储在独立的作用域中。因此直接打印 config 时看不到它们
```yaml
hydra:
  run:
    dir: ${experiment.exp_dir}/hydra
```

**inference/tp2.yaml**
- **大语言模型(llm)配置** 
  - 大型语言模型（LLM）系统中，Model（模型）和Tokenizer（分词器）是两个核心组件
    - `Model`是整个系统的“大脑”, 能够基于输入数据生成文本或其他输出（包含了从训练中学到的语言规律、知识表示和推理能力）
      - 能根据给定的提示词，通过计算概率分布预测下一个最可能的单词序列，最终形成连贯的自然语言响应
    - `tokenizer` 将原始文本转换为模型能理解的数字编码形式
      - 通常采用子词分割算法（如BPE或WordPiece），将文本切分为有意义的子单元并为每个子词分配唯一ID
      - 对文本标准化预处理: 包括清洗特殊字符、统一大小写、添加开始/结束标记等操作，确保输入符合模型要求
      - 通过参数如max_seq_len限制最大序列长度，平衡内存占用与模型性能
      - 支持正向编码（文本→ID序列）和反向解码（ID序列→文本），便于调试和可视化中间结果
  - **llm** 定义了模型本身的属性，如路径、并行策略、资源利用率等（相当于“发动机”设置）,决定了模型的加载与运行，主要涉及性能优化和硬件调度
    - `model` 
    - `tokenizer`
    - `trust_remote_code` 允许动态下载并执行模型库中的自定义Python函数
    - `tensor_parallel_size` 张量并行度, 将巨型矩阵运算拆分到多块GPU上同时计算
    - `pipeline_parallel_size` 流水线并行度=1。不启用跨设备的阶段级流水线（即完整的前向传播仍在同一设备完成）。适合小规模部署或单卡推理
    - `gpu_memory_utilization` GPU显存利用率90%。系统会尽量占满每张卡90%的显存以提高效率，剩下10%作为缓冲区防止溢出
    - `seed` 随机种子固定为1234。确保多次运行结果一致（可复现性），对调试非常友好；关闭随机性则设为负数（如 -1）
  - **generate** 指定具体的生成任务，包括输入提示词、采样参数等（相当于“驾驶模式”设置）
    - `prompts` 输入提示词 
    - `sampling` 采样策略
      - `top_p` 累积概率阈值法筛选候选集，保留头部累计概率≥top_p的所有token作为有效候选池
        - 数值越小越保守精准，越大则创意发散
        - top_p=1.0 → 包括全部词汇表（等价于关闭此过滤）
        - top_p<1.0 → 动态截断小概率尾部词汇，减少离群值干扰
      - `top_k` 限制考虑范围到前K个最高概率的token
        - k=1 → 仅允许排名第一的那个token参与后续计算
        - k>1 → 允许更多token作为备选项，提升潜在创意空间
      - `temperature` 温度系数,调节随机性强度
        - 接近0时趋于确定值
        - 大于1时随机性增强
        - ∞ 完全随机采样
      - `do_sample` 是否启用随机采样模式
        - 参数为 true ,上述三个参数才有意义
        - 参数为 false , 无论其他参数如何设置，整个系统仍然运行在非采样模式下，严格按照概率最大的单一路径推进


```yaml
llm:
  model: /home/gitlab-runner/data/DeepSeek-R1-Distill-Qwen-7B
  tokenizer: /home/gitlab-runner/data/DeepSeek-R1-Distill-Qwen-7B
  trust_remote_code: true
  tensor_parallel_size: 2
  pipeline_parallel_size: 1
  gpu_memory_utilization: 0.9
  seed: 1234

generate:
  prompts: [
    "The president of the United States",
    "The capital of France",
  ]
  sampling:
    top_p: 0.1
    top_k: 1
    temperature: 0.1
```
