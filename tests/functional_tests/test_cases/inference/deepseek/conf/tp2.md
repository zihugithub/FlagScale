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
  - `NCCL_ALGO` 指定多机通信拓扑采用的算法（减少带宽瓶颈）
  - `NVTE_APPLY_QK_LAYER_SCALING: 0` 控制 NeMo VTE插件查询键层缩放功能（0 关闭）
  - `NVTE_ALLOW_NONDETERMINISTIC_ALGO: 0` 强制确定性计算以保证可复现性
  - `NVTE_FLASH_ATTN: 0` 禁用Flash Attention特性（因兼容性或其他原因）
  - `NVTE_FUSED_ATTN: 0` 不融合注意力头的操作
  - `CUDNN_BENCHMARK: "false"` 避免首次运行时的性能基准测试开销
  - `CUDNN_DETERMINISTIC: "true"` 确保卷积操作结果完全确定性再现

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

顶层指令：立即执行此配置文件描述的任务
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
    - `gpu_memory_utilization` GPU显存利用率90%。系统会尽量占满每张卡90%的显存以提高效率，剩下10%作为缓冲区防止溢出（OOM Killer
    - `seed` 随机种子固定为1234。确保多次运行结果一致（可复现性），对调试非常友好；关闭随机性则设为负数（如 -1）
  - **generate** 指定具体的生成任务，包括输入提示词、采样参数等（相当于“驾驶模式”设置）
    - `prompts` 输入提示词 
    - `sampling` 采样策略
      - `top_p` 核心采样阈值=0.1。仅保留概率累计超过10%的最高候选词参与下一步选择。数值越小越保守精准；越大则创意发散
      - `temperature` 温度系数=0.1（低温模式）。压低所有词汇的概率分布熵值，使高分词几乎必选。接近0时趋于确定性解码（Argmax）；大于1时随机性增强

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
    temperature: 0.1
```
