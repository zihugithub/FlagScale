---
name: flagscale-dev
description: >
  FlagScale 项目编码辅助。当开发者需要：(1) 理解 FlagScale 架构和代码流程，
  (2) 添加新模型支持（train/serve/inference），(3) 修改 Runner/Launcher/Backend，
  (4) 编写或调试 Hydra 配置，(5) 编写测试，(6) 调试分布式训练/推理问题，
  (7) 贡献代码或 review PR 时激活此 Skill。
---

# FlagScale 编码辅助

## 项目概览

FlagScale 是大模型全生命周期工具包，统一管理 train / serve / inference / auto_tune 任务。
底层集成 Megatron-LM（训练）和 vLLM（推理服务），通过 Hydra 配置驱动一切。

- **仓库**：`https://github.com/FlagOpen/FlagScale`
- **Python 包**：`flagscale`，CLI 入口 `flagscale`
- **配置**：Hydra + OmegaConf（YAML）
- **支持模型**：Aquila, DeepSeek-R1/V3, Qwen2/2.5/3, LLaMA2/3, Mixtral, RWKV, Flux, LLaVA, MiniCPM, Emu3, Pi0 等 30+ 模型

## 目录结构

```
FlagScale/
├── flagscale/              # 核心 Python 包
│   ├── cli.py              # CLI 入口 (flagscale train/serve/inference)
│   ├── run.py              # 任务调度：解析配置 → RunnerFactory → runner.run()
│   ├── runner/             # 🔑 核心调度层
│   │   ├── runner_base.py          # RunnerBase 基类（新架构）
│   │   ├── runner_base_legacy.py   # 旧 RunnerBase（serve 仍使用）
│   │   ├── runner_train.py         # 训练 Runner
│   │   ├── runner_serve.py         # 推理服务 Runner（vLLM）
│   │   ├── runner_inference.py     # 离线推理 Runner
│   │   ├── runner_factory.py       # 工厂：根据 task.type 分发 Runner
│   │   ├── launcher/               # 启动器：local / SSH / Cloud
│   │   ├── backend/                # 后端抽象：Megatron, vLLM 等
│   │   └── auto_tuner/             # 自动调参（并行策略搜索）
│   ├── serve/              # Serve 专用模块
│   │   ├── core.py                 # 多模型部署核心
│   │   ├── engine.py               # 推理引擎管理
│   │   ├── run_serve.py            # Serve 入口
│   │   └── args_mapping/           # 模型参数映射
│   ├── train/              # 训练入口脚本
│   ├── models/megatron/    # Megatron 模型适配层
│   └── patches/            # 对上游 Megatron/vLLM 的补丁
├── megatron/               # Megatron-LM（训练后端，有 FlagScale 补丁）
├── vllm/                   # vLLM（推理后端，有 FlagScale 补丁）
├── examples/               # 模型配置示例（每模型一目录）
│   └── <model>/conf/       # Hydra 配置：train.yaml, serve.yaml 等
├── tests/                  # 测试
│   └── unit_tests/
└── docs/                   # 文档
```

## 核心代码流程

### 入口：CLI → run.py

```
flagscale train qwen3 --config examples/qwen3/conf/train.yaml
    ↓
cli.py: 解析子命令（train/serve/inference）
    ↓
run.py: load_config() → RunnerFactory.create(config) → runner.run()
```

### Runner 体系（最核心）

```
RunnerBase (runner_base.py)
├── RunnerTrain (runner_train.py)     ← 分布式训练
├── RunnerInference (runner_inference.py) ← 离线推理
└── RunnerServe (runner_serve.py)     ← vLLM 推理服务（继承 legacy base）

RunnerBase 职责：
  - 解析 hostfile → 构建节点拓扑
  - 选择 Launcher（local/SSH/Cloud）
  - 选择 Backend（Megatron/vLLM）
  - 生成启动命令 → 分发到各节点执行
  - 管理任务生命周期（启动/停止/监控）
```

### Hydra 配置结构

```yaml
# examples/qwen3/conf/train.yaml
defaults:
  - _self_
  - train: 0_6b          # 引用 train/0_6b.yaml

experiment:
  exp_name: qwen3_0_6b
  task:
    type: train           # → RunnerFactory 选择 RunnerTrain
    backend: megatron     # → 使用 Megatron 后端
  runner:
    hostfile: /path/to/hostfile
    ...
  model:
    ...                   # Megatron 模型参数
```

## 常用开发任务

### 添加新模型支持

1. 读 [references/add-model.md](references/add-model.md) 了解完整流程
2. 核心步骤：创建 `examples/<model>/conf/` 配置 → 添加模型适配代码 → 测试

### 修改 Runner / Launcher / Backend

- Runner 新功能：继承 `RunnerBase`，重写 `_run()` / `_stop()`
- Launcher 新类型：在 `runner/launcher/` 添加，注册到 Runner
- Backend 新后端：在 `runner/backend/` 添加，实现 `BackendBase` 接口

### 修改 Serve 模块

- 参数映射：`serve/args_mapping/<model>.py`
- 多模型部署：`serve/core.py` → `ServeCoreManager`
- 引擎管理：`serve/engine.py`

### 编写 Hydra 配置

- 用 `defaults` 实现配置组合（不要平铺所有参数）
- `experiment.task.type` 决定 Runner 类型
- `experiment.task.backend` 决定后端
- 用 `OmegaConf.to_container(config, resolve=True)` 获取最终值

### 编写测试

- 测试目录：`tests/unit_tests/`
- 运行：`pytest tests/unit_tests/ -v`
- 新模型至少测试：配置加载、参数映射、基本前向推理

### 调试分布式训练问题

1. 检查 hostfile 格式（`hostname slots=N`）
2. 检查 SSH 免密配置
3. 查看日志：`outputs/<exp_name>/train_logs/`
4. 常见问题：NCCL 超时 → 检查网络/防火墙；OOM → 调整并行策略
5. 使用 auto_tuner 自动搜索最优并行配置

## 代码风格

- Python 3.8+，使用 type hints
- 遵循 PEP 8，行宽 120
- 配置用 Hydra/OmegaConf，不要硬编码
- 日志用 `flagscale.runner.utils.logger`
- 新文件需要 Apache 2.0 License header

## PR 贡献规范

- Branch 命名：`feature/<desc>` / `fix/<desc>`
- Commit message：清晰描述改动
- PR 需要 MAINTAINERS 中对应模块 owner review
- 参考 `MAINTAINERS.md` 了解模块负责人

## 参考文档

- **添加新模型**：[references/add-model.md](references/add-model.md)
- **架构详解**：[references/architecture.md](references/architecture.md)
- **FlagScale 官方文档**：`docs/` 目录
