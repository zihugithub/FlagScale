# FlagScale Docker 镜像构建体系
为灵活适配多样化业务场景，FlagScale 提供模块化、可扩展的 Dockerfile 构建方案，支持多环境独立部署与集成化开发。各模块职责分明，便于维护与扩展。

## 1. 基础镜像构建

| 芯片类型 | 功能定位 | 配置文件 | 核心特性 |
|---------|----------|-------------|---------------|
| Nvidia | 提供标准化的 CUDA/cuDNN/Python 运行环境 | docker/base/Dockerfile.cuda.base |  1) 预装指定版本的 NVIDIA CUDA（如 12.8.1）和 cuDNN（如 9.15.1）<br> 2) 集成 Python 3.12 及常用系统工具链<br> 3) 支持代理配置与缓存清理优化 |

**技术栈明细：**
| 组件 | 版本信息 | 作用说明|
|------|----------|--------|
| sccache | 0.12.0 | 使用编译缓存加速编译
| OpenMPI | 5.0.9 | 1. 多机多卡协同加速大规模模型并行训练<br>2. 自动触发跨多个容器的集成测试，验证分布式系统的容错性和性能<br>3. 确保开发人员本地环境和生产环境的 MPI 版本、编译参数完全一致
| miniconda | latest | 1. 快速搭建PyTorch 等库的开发环境<br>2. 在流水线中构建可复现的测试环境<br>3. 每个服务独立运行在自己的 Conda 环境中，降低耦合度
| cuDNN | 9.15.1 | 1. 利用 GPU 加速科学计算任务<br> 2. 在容器中部署基于 CuDNN 加速的模型（如Transformer）<br> 3. 在支持 NVIDIA GPU 的边缘设备上运行轻量级的 AI 推理服务

**使用方式：**
```sh
# 构建 FlagScale 基础镜像
# 支持 CUDA 12.8.1、cuDNN 9.15.1、Python 3.12
docker build \
   # 设置 HTTP 代理服务器地址和端口号
   --build-arg http_proxy=http://x.x.x.x:xxxx
   --build-arg https_proxy=http://x.x.x.x:xxxx
   # 指定 CUDA 版本号
   --build-arg CUDA_VERSION=12.8.1 \
   -f docker/base/dockerfile.cuda.base \
   -t flagscale:cuda12.8.1-cudnn9.15.1-python3.12-base .
```

## 2. 推理服务镜像
**针对不同推理框架需求，提供定制化镜像：**

| 芯片类型 | 支持后端 | 配置文件 |  适用场景 |
|---------|----------|---------|----------|
| Nvidia  | VLLM | docker/inference/Dockerfile.cuda.vllm | 基于 VLLM 构建满足 Nvidia 的推理镜像
| MetaX   | VLLM | docker/inference/Dockerfile.metax.vllm | 基于 VLLM 构建满足 Metax 的推理镜像


**技术栈明细：**
| 组件 | 版本信息 | 作用说明|
|------|----------|--------|
| VLLM | unpath同步 | LLM 推理引擎核心框架 |
|PyTorch | 2.7.1+cu128 | 深度学习基础运行环境，集成 CUDA 12.8 编译器与 cuDNN 9.15.1 加速库 |
| FlagGems | 3.0 | 自研算子库 |

**使用方法：**
```sh
# 构建 FlagScale 推理镜像
docker build \
   # # 设置 HTTP/HTTPS 代理服务器地址和端口号
   --build-arg http_proxy=http://x.x.x.x:xxxx \
   --build-arg https_proxy=http://x.x.x.x:xxxx \
   # 基础镜像名称及版本信息
   --build-arg BASE_IMAGE=flagscale:cuda12.8.1-cudnn9.15.1-python3.12-base \
   # 指定 FlagScale 项目的仓库 URL
   --build-arg FLAGSCALE_REPO=https://github.com/your-repo/FlagScale.git \
   # 选择要克隆的分支名称
   --build-arg FLAGSCALE_BRANCH=your-branch \
   # 选择要切换的commit
   --build-arg FLAGSCALE_COMMIT=your-commit \
   # PyTorch 版本号（包含 CUDA 12.8 支持）
   --build-arg PYTORCH_VER="2.7.1+cu128" \
   # TorchAudio 版本号（与 PyTorch 兼容）
   --build-arg TORCHAUDIO_VER="2.7.1+cu128" \
   # TorchVision 版本号（用于图像处理任务）
   --build-arg TORCHVISION_VER="0.22.1+cu128" \
   # 额外的 PyPI 索引地址，用于安装特定版本的依赖项
   --build-arg EXTRA_INDEX="https://download.pytorch.org/whl/cu128" \
   # 指定使用的 Dockerfile 路径
   -f docker/inference/Dockerfile.cuda.vllm \
   # 目标镜像标签命名规则：<CUDA 版本>-<cuDNN 版本>-<Python 版本>-<PyTorch 版本>-<时间戳>-<用途>
   -t flagscale:cuda12.8.1-cudnn9.15.1-python3.12-torch2.7.1-time2512031818-inference .

# 运行容器
docker run -itd --gpus all --shm-size=500g \
   --name Inference2512031818 \
   --hostname flagscale_inference \
   -v /data/flagscale_cicd/docker/docker_build/docker_data:/home/gitlab-runner/data \
   -v /data/flagscale_cicd/docker/docker_build/docker_tokenizers:/home/gitlab-runner/tokenizers flagscale:cuda12.8.1-cudnn9.15.1-python3.12-torch2.7.1-time2512031818-inference

# 进入容器
docker exec -it Inference2512031818 bash
```

## 3. 训练任务镜像
专为大规模分布式训练设计：

| 芯片类型 | 支持后端 | 配置文件 |  适用场景 |
|---------|----------|---------|----------|
| Nvidia  | megatron | docker/train/Dockerfile.cuda.megatron | 基于 megatron 构建满足 Nvidia 的训练镜像

**技术栈明细：**
| 组件 | 版本信息 | 作用说明|
|------|----------|--------|
| megatron | unpath同步 | 大规模分布式训练框架，支持万亿参数模型并行训练 |
|PyTorch | 2.7.1+cu128 | 深度学习基础运行环境，集成 CUDA 12.8 编译器与 cuDNN 9.15.1 加速库 |
| TransformerEngine | e9a5fa4e | 高效 transformer 计算引擎，提供混合精度训练与动态张量并行支持 |
| Flash Attention | 2.8.0.post2 | 显存优化的注意力机制，通过分块计算降低内存占用 |
| GROUPED GEMM | 1.1.4.post6 | 分组通用矩阵乘法库，专为稀疏/结构化张量设计，提升 MoE 模型推理吞吐量 |
| APEX |  0.1 | NVIDIA 官方混合精度训练工具链，支持自动损失缩放与梯度裁剪 |

**使用方式：**
```sh
# 构建 FlagScale 训练镜像
# 支持 CUDA 12.8.1、cuDNN 9.15.1、Python 3.12 和 PyTorch 2.7.1+cu128
docker build \
   # 设置 HTTP/HTTPS 代理服务器地址和端口号
   --build-arg http_proxy=http://x.x.x.x:xxxx \
   --build-arg https_proxy=http://x.x.x.x:xxxx \
   # 基础镜像名称及版本信息
   --build-arg BASE_IMAGE=flagscale:cuda12.8.1-cudnn9.15.1-python3.12-base \
   # 指定 FlagScale 项目的仓库 URL
   --build-arg FLAGSCALE_REPO=https://github.com/your-repo/FlagScale.git \
   # 选择要克隆的分支名称
   --build-arg FLAGSCALE_BRANCH=your-branch \
   # 选择要切换的commit
   --build-arg FLAGSCALE_COMMIT=your-commit \
   # PyTorch 版本号（包含 CUDA 12.8 支持）
   --build-arg PYTORCH_VER="2.7.1+cu128" \
   # TorchAudio 版本号（与 PyTorch 兼容）
   --build-arg TORCHAUDIO_VER="2.7.1+cu128" \
   # TorchVision 版本号（用于图像处理任务）
   --build-arg TORCHVISION_VER="0.22.1+cu128" \
   # 额外的 PyPI 索引地址，用于安装特定版本的依赖项
   --build-arg EXTRA_INDEX="https://download.pytorch.org/whl/cu128" \
   # Transformer Engine 提交哈希值（请参考官方文档确保与所选 PyTorch 版本匹配）
   --build-arg TRANSFORMER_ENGINE_COMMIT="e9a5fa4e" \
   # Flash Attention 库的版本号（请参考官方文档确保与所选 PyTorch 版本匹配）
   --build-arg FLASH_ATTN_VERSION="2.8.0.post2" \
   # Grouped Gemmi 库的版本号（请参考官方文档确保与所选 PyTorch 版本匹配）
   --build-arg GROUPED_GEMM_VERSION="1.1.4.post6" \
   # 指定使用的 Dockerfile 路径
   -f docker/train/Dockerfile.cuda.megatron \
   # 目标镜像标签命名规则：<CUDA 版本>-<cuDNN 版本>-<Python 版本>-<PyTorch 版本>-<时间戳>-<用途>
   -t flagscale:cuda12.8.1-cudnn9.15.1-python3.12-torch2.7.1-time2512031818-train .

# 运行容器
docker run -itd --gpus all --shm-size=500g \
   --name Train2512031818 \
   --hostname flagscale_train \
   -v /data/flagscale_cicd/docker/docker_build/docker_data:/home/gitlab-runner/data \
   -v /data/flagscale_cicd/docker/docker_build/docker_tokenizers:/home/gitlab-runner/tokenizers flagscale:cuda12.8.1-cudnn9.15.1-python3.12-torch2.7.1-time2512031818-train

# 进入容器
docker exec -it Train2512031818 bash
```

## 4. 强化学习（RL）镜像
面向 RL 算法开发的专用环境：

- 配置文件：docker/rl/Dockerfile.cuda.verl
- 特色组件：
   - Verl 框架预安装（含 Flash Attention 等依赖）
   - 动态资源调度支持
   - 兼容 Grouped Gemmi 算子库

**使用方式：**
```sh
# 构建 FlagScale 强化学习镜像
# 支持 CUDA 12.8.1、cuDNN 9.15.1、Python 3.10 和 PyTorch
docker build \
   # 设置 HTTP/HTTPS 代理服务器地址和端口号
   --build-arg http_proxy=http://x.x.x.x:xxxx \
   --build-arg https_proxy=http://x.x.x.x:xxxx \
   # 基础镜像名称及版本信息
   --build-arg BASE_IMAGE=flagscale:cuda12.8.1-cudnn9.15.1-python3.12-base \
   # 指定 FlagScale 项目的仓库 URL
   --build-arg FLAGSCALE_REPO=https://github.com/your-repo/FlagScale.git \
   # 选择要克隆的分支名称
   --build-arg FLAGSCALE_BRANCH=your-branch \
   # 选择要切换的commit
   --build-arg FLAGSCALE_COMMIT=your-commit \
   # 指定使用的 Dockerfile 路径
   -f docker/rl/Dockerfile.cuda.verl \
   # 目标镜像标签命名规则：<CUDA 版本>-<cuDNN 版本>-<Python 版本>-<PyTorch 版本>-<时间戳>-<用途>
   -t flagscale:cuda12.8.1-cudnn9.15.1-python3.10-torch2.6.0-time2512031818-rl .

# 运行容器
docker run -itd --gpus all --shm-size=500g \
   --name RL2512031818 \
   --hostname flagscale_train \
   -v /data/flagscale_cicd/docker/docker_build/docker_data:/home/gitlab-runner/data \
   -v /data/flagscale_cicd/docker/docker_build/docker_tokenizers:/home/gitlab-runner/tokenizers flagscale:cuda12.8.1-cudnn9.15.1-python3.10-torch2.6.0-time2512031818-rl

# 进入容器
docker exec -it RL2512031818 bash
```


## 5. 全功能集成镜像
一键式获取完整开发环境：

- 配置文件：docker/Dockerfile.cuda
- 优势：
   - 聚合 inference/train/RL 三大环境

**使用方式：**
```sh
docker build \
   # 设置 HTTP/HTTPS 代理服务器地址和端口号
   --build-arg http_proxy=http://x.x.x.x:xxxx \
   --build-arg https_proxy=http://x.x.x.x:xxxx \
   # 基础镜像名称及版本信息
   --build-arg FLAGSCALE_BASE=flagscale:cuda12.8.1-cudnn9.15.1-python3.12-base \
   # 推理镜像名称及版本信息
   --build-arg FLAGSCALE_INFERENCE=flagscale:cuda12.8.1-cudnn9.15.1-python3.12-torch2.7.1-time2512031818-inference \
   # 训练镜像名称及版本信息
   --build-arg FLAGSCALE_TRAIN=flagscale:cuda12.8.1-cudnn9.15.1-python3.12-torch2.7.1-time2512031818-train \
   # 强化学习镜像名称及版本信息
   --build-arg FLAGSCALE_RL=flagscale:cuda12.8.1-cudnn9.15.1-python3.10-torch2.6.0-time2512031818-rl \
   # 指定使用的 Dockerfile 路径
   -f docker/Dockerfile.cuda \
   # 目标镜像标签命名规则：<CUDA 版本>-<cuDNN 版本>-<Python 版本>-<PyTorch 版本>-<时间戳>
   -t flagscale:cuda12.8.1-cudnn9.15.1-python3.10-torch2.6.0-time2512031818 .

```

## 6. SSH 免密登录增强
**安全警告与操作建议**
- 本镜像构建方式存在安全隐患，仅限内部开发环境使用，严禁外泄已构建的镜像文件
- 为避免相同密钥导致的环境污染风险，强烈建议针对不同任务每次重建新镜像
```sh
docker build \
   --build-arg BASE_IMAGE=flagscale:cuda12.4.1-cudnn9.5.0-python3.12-torch2.5.1-time2503251131 
   --build-arg SSH_PORT=22 
   -f docker/Dockerfile.ssh 
   -t flagscale:cuda12.4.1-cudnn9.5.0-python3.12-torch2.5.1-time2503251131-ssh .
```
