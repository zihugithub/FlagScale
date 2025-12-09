# 环境安装指南
本文介绍如何使用 Conda 隔离特性构建不同用途的环境。请确保已安装 `Miniconda` 后再执行以下操作。

## 1. 安装 Miniconda（如尚未安装）
```sh
# 创建安装目录并下载最新 Miniconda
mkdir -p ~/miniconda3 && \
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \

# 静默安装并清理安装包
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
rm -rf ~/miniconda3/miniconda.sh && \

# 初始化 Conda 并配置默认设置
~/miniconda3/bin/conda init bash && \
~/miniconda3/bin/conda config --set auto_activate_base false && \
~/miniconda3/bin/conda config --set default_python 3.12

```

**提示:** 安装完成后建议重启终端或运行 source ~/miniconda3/etc/profile.d/conda.sh 使 Conda 生效

# 2. 训练环境安装（Megatron）

```bash
bash ./install/train/install-nvidia-megatron.sh \
    --env train \                                              # 必须：指定环境名称
    --torch-ver "2.7.1+cu128" \                                # PyTorch 版本
    --torchaudio-ver "2.7.1+cu128" \                           # TorchAudio 版本
    --torchvision-ver "0.22.1+cu128" \                         # TorchVision 版本
    --extra-index "https://download.pytorch.org/whl/cu128" \   # 自定义 PyTorch 源
    --flash-attn-ver "2.8.0.post2" \                           # Flash Attention 版本
    --group-gemm-ver "1.1.4.post6" \                           # Group GEMM 版本
    --transformer-engine-commit "e9a5fa4e"                     # Transformer Engine 提交ID
```

# 3. 推理环境安装（vLLM）

```bash
bash ./install/inference/install-nvidia-vllm.sh \
    --env inference \                                          # 必须：指定环境名称
    --llama-cpp-backend "cpu" \                                # LlamaCPP 后端
    --omni_infer "0" \                                         # Omni Inference 开关
    --torch-ver "2.7.1+cu128" \                                # PyTorch 版本
    --torchaudio-ver "2.7.1+cu128" \                           # TorchAudio 版本
    --torchvision-ver "0.22.1+cu128" \                         # TorchVision 版本
    --extra-index "https://download.pytorch.org/whl/cu128"     # 自定义 PyTorch 源
```

# 4. 强化学习环境安装（Verl）

```bash
bash ./install/rl/install-nvidia-verl.sh \
    --env RL                                                   # 必须：指定环境名称
```

# 5. MetaX 推理环境安装

```bash
bash ./install/inference/install-metax-vllm.sh \
    --env "inference" \                                        # 必须：指定环境名称
    --torch-ver "2.6.0+metax3.0.0.3" \                         # MetaX 兼容版 PyTorch
    --torchaudio-ver "2.4.1+metax3.0.0.3" \                    # MetaX 兼容版 TorchAudio
    --torchvision-ver "0.15.1+metax3.0.0.3"                    # MetaX 兼容版 TorchVision
```

# torch版本 与 其它软件版本的对应关系
| torch | TorchAudio | TorchVision |
|-------|------------|-------------|
| 2.6.0 | 2.6.0      | 0.21.0      |
| 2.7.0 | 2.7.0      | 0.22.0      |
| 2.7.1 | 2.7.1      | 0.22.1      |
| 2.8.0 | 2.8.0      | 0.23.0      |
| 2.9.0 | 2.9.0      | 0.24.0      |

注：更多touch版本与TorchVision版本对应关系见：https://github.com/pytorch/vision#installation

| torch | vllm | flash-attn | group-gemm | transformer-engine | verl |
|-------|------|------------|------------|--------------------|------|
| 2.6.0 | 0.8.6.dev0+gba41cc90e.d20250524 | 2.7.3 | 1.1.2 | 5bee81e | |
| 2.7.0 | 0.9.2.dev0+gb6553be1b.d20250626 | 2.8.0.post2 | 1.1.2 | 5bee81e | |
| 2.7.1 | 0.10.1.dev0+g6d8d0a24c.d20250925 | 2.8.0.post2 | 1.1.2 | e9a5fa4e | |
| 2.8.0 | 0.11.0 | 
| 2.9.0 | xxxx | xxxxxxxxxx | 1.1.4.post6 | 29537c96          |       |

