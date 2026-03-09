# FlagScale 架构详解

## 分层架构

```
┌────────────────────────────────────────────────┐
│  CLI Layer (cli.py)                            │
│  flagscale train / serve / inference           │
├────────────────────────────────────────────────┤
│  Config Layer (Hydra + OmegaConf)              │
│  examples/<model>/conf/*.yaml                  │
├────────────────────────────────────────────────┤
│  Orchestration Layer (run.py)                  │
│  load_config → RunnerFactory → runner.run()    │
├────────────────────────────────────────────────┤
│  Runner Layer                                  │
│  ┌──────────┬──────────────┬──────────────┐   │
│  │RunnerTrain│RunnerServe  │RunnerInference│   │
│  └────┬─────┴──────┬──────┴──────┬────────┘   │
│       │            │             │              │
│  ┌────┴────┐ ┌─────┴─────┐ ┌────┴────┐       │
│  │Launcher │ │  Backend   │ │AutoTuner│       │
│  │local/SSH│ │Megatron/   │ │并行策略  │       │
│  │/Cloud   │ │vLLM        │ │搜索     │       │
│  └─────────┘ └───────────┘ └─────────┘       │
├────────────────────────────────────────────────┤
│  Execution Layer                               │
│  Megatron-LM / vLLM (带 FlagScale 补丁)        │
└────────────────────────────────────────────────┘
```

## 核心类关系

### RunnerBase（新架构，runner_base.py）

```python
class RunnerBase:
    """所有 Runner 的基类"""
    
    def __init__(self, config):
        self.config = config
        self.launcher = None    # 由子类初始化
        self.backend = None     # 由子类初始化
    
    def run(self):
        """主流程：setup → prepare → launch → monitor"""
        self._setup()           # 解析 hostfile、创建日志目录
        self._prepare()         # 生成启动命令
        self._launch()          # 通过 Launcher 分发执行
        self._monitor()         # 监控任务状态
    
    def stop(self):
        """停止任务"""
        self._stop()
```

### RunnerBase Legacy（runner_base_legacy.py）

RunnerServe 仍使用旧版基类，包含更多 serve 特有逻辑：
- `JobStatus` 枚举：STARTING / RUNNING / STOPPED / ERROR
- 进程管理、端口分配、健康检查

### RunnerTrain（runner_train.py）

```python
class RunnerTrain(RunnerBase):
    """分布式训练 Runner"""
    
    # 关键方法：
    def _generate_run_script(self):
        """生成 torchrun / 多节点启动脚本"""
        # 处理 TP/PP/DP 并行策略
        # 生成 hostfile、环境变量
    
    def _run(self):
        """执行训练"""
        # SSH 到各节点 → 执行训练脚本
```

### RunnerServe（runner_serve.py）

```python
class RunnerServe(RunnerBaseLegacy):
    """vLLM 推理服务 Runner"""
    
    # 关键函数（模块级）：
    # _get_args_vllm(config) → 将 Hydra config 转为 vLLM CLI 参数
    # _reset_serve_port(config) → 端口管理
    
    # 支持：
    # - 单模型 / 多模型部署
    # - Ray 分布式推理
    # - 自动端口分配
```

### Launcher 体系

```
runner/launcher/
├── local_launcher.py    # 本地启动（subprocess）
├── ssh_launcher.py      # SSH 远程启动（多节点）
└── cloud_launcher.py    # 云平台启动（K8s 等）
```

Launcher 负责将生成的命令分发到目标节点执行。

### Backend 体系

```
runner/backend/
├── backend_base.py      # 后端抽象基类
├── megatron.py          # Megatron-LM 后端
└── vllm.py              # vLLM 后端
```

Backend 负责：
- 参数转换（FlagScale config → 后端 CLI args）
- 环境准备（NCCL 配置、CUDA 可见设备等）
- 启动命令生成

## 配置系统

### Hydra 配置组合

FlagScale 使用 Hydra 的 `defaults` 实现配置组合：

```yaml
# train.yaml (主配置)
defaults:
  - _self_
  - train: 0_6b      # → 加载 train/0_6b.yaml 并合并

experiment:
  task:
    type: train       # 决定使用哪个 Runner
    backend: megatron # 决定使用哪个 Backend
```

### 关键配置路径

| 配置路径 | 作用 |
|---------|------|
| `experiment.task.type` | Runner 类型：train / serve / inference |
| `experiment.task.backend` | 后端：megatron / vllm |
| `experiment.runner.hostfile` | 节点列表 |
| `experiment.runner.deploy.port` | Serve 端口 |
| `experiment.model.*` | 模型参数（传给后端） |

### 配置覆盖

命令行可覆盖任何配置：
```bash
flagscale train qwen3 --config ... experiment.model.num_layers=12
```

## Serve 模块详解

### 多模型部署（core.py）

`ServeCoreManager` 支持在同一进程中部署多个模型：

```python
class ServeCoreManager:
    def __init__(self):
        self.engines = {}   # model_name → engine
    
    def add_model(self, name, config):
        engine = create_engine(config)
        self.engines[name] = engine
    
    async def generate(self, model, prompt, params):
        return await self.engines[model].generate(prompt, params)
```

### 参数映射（args_mapping/）

每个模型一个映射文件，将 FlagScale 统一配置转为 vLLM 特定参数：

```python
# args_mapping/qwen3.py
ARGS_MAPPING = {
    "model": "model",
    "tensor_parallel_size": "tensor-parallel-size",
    "max_model_len": "max-model-len",
}
```

## 测试架构

```
tests/
└── unit_tests/
    ├── test_runner.py         # Runner 逻辑测试
    ├── test_config.py         # 配置加载测试
    └── test_args_mapping.py   # 参数映射测试
```

运行测试：
```bash
pytest tests/unit_tests/ -v
pytest tests/unit_tests/test_runner.py -k "test_train" -v
```

## 插件系统

FlagScale 通过 Git Submodule 管理外部依赖：
- `megatron/` → Megatron-LM（带 FlagScale 补丁）
- `vllm/` → vLLM（带 FlagScale 补丁）

补丁在 `flagscale/patches/` 下管理，通过 setup 时 apply。
