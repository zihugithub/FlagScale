# 添加新模型支持

## 总览

FlagScale 支持三种任务类型，每种的模型接入流程不同：

| 任务 | 后端 | 主要工作 |
|------|------|---------|
| train | Megatron-LM | 模型定义 + 训练配置 |
| serve | vLLM | 参数映射 + 部署配置 |
| inference | Megatron-LM | 推理脚本 + 配置 |

## 一、添加训练支持（train）

### 1. 创建配置目录

```bash
mkdir -p examples/<model>/conf/train
```

### 2. 编写主配置 `examples/<model>/conf/train.yaml`

```yaml
defaults:
  - _self_
  - train: <size>        # 如 0_6b, 7b, 70b

experiment:
  exp_name: <model>_<size>
  task:
    type: train
    backend: megatron
  runner:
    hostfile: ???         # 必须由用户指定
    # 其他 runner 参数
```

### 3. 编写尺寸配置 `examples/<model>/conf/train/<size>.yaml`

包含模型特定参数：hidden_size, num_layers, num_attention_heads 等。

参考已有模型：`examples/qwen3/conf/train/0_6b.yaml`

### 4. 如需模型适配

在 `flagscale/models/megatron/` 下添加适配代码，处理：
- 模型架构差异（如 GQA、MoE、RoPE 变体）
- 分词器映射
- 权重转换

### 5. 测试

```bash
# 单机小规模测试
flagscale train <model> --config examples/<model>/conf/train.yaml \
  experiment.runner.hostfile=null
```

## 二、添加 Serve 支持

### 1. 创建参数映射（如果 vLLM 原生不支持）

在 `flagscale/serve/args_mapping/` 下添加 `<model>.py`：

```python
# flagscale/serve/args_mapping/<model>.py

ARGS_MAPPING = {
    "model": "model",
    "tensor_parallel_size": "tensor-parallel-size",
    # FlagScale 配置 key → vLLM CLI 参数
}

def transform_config(config):
    """对配置做模型特定的变换"""
    # 可选：处理特殊参数
    return config
```

### 2. 编写 serve 配置

```yaml
# examples/<model>/conf/serve.yaml
defaults:
  - _self_
  - serve: <size>

experiment:
  exp_name: <model>_serve
  task:
    type: serve
    backend: vllm
  runner:
    deploy:
      port: 30000
```

### 3. 测试

```bash
flagscale serve <model> --config examples/<model>/conf/serve.yaml
# 验证 API
curl http://localhost:30000/v1/models
curl http://localhost:30000/v1/chat/completions \
  -d '{"model":"<model>","messages":[{"role":"user","content":"hello"}]}'
```

## 三、添加 Inference 支持

### 1. 编写推理脚本

在 `flagscale/train/` 或单独目录下添加推理入口。

### 2. 编写配置

```yaml
# examples/<model>/conf/inference.yaml
defaults:
  - _self_
  - inference: <size>

experiment:
  exp_name: <model>_inference
  task:
    type: inference
    backend: megatron
```

## Checklist

添加新模型时确保：

- [ ] 配置文件结构正确（defaults 引用、task type/backend）
- [ ] 至少一个尺寸配置可工作
- [ ] hostfile 设为 `???` 强制用户指定（或 null 表示单机）
- [ ] 添加到 `examples/` 目录
- [ ] 编写基本测试
- [ ] 更新 MAINTAINERS.md（如果你是该模型 owner）
