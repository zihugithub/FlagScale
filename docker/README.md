# FlagScale Docker Image Build System
To flexibly adapt to diverse business scenarios, FlagScale offers a modular and extensible Dockerfile build solution that supports independent deployment across multiple environments and integrated development. Each module has clear responsibilities, facilitating maintenance and expansion.

## 1. Base Image Construction
| Chip Type	| Function Positioning	| Configuration| 	Core Features |
|---------|----------|-------------|---------------|
| Nvidia	| Provides standardized CUDA/cuDNN/Python runtime environment	| docker/base/Dockerfile.cuda.base | 1) Pre-install specified versions of NVIDIA CUDA (e.g., 12.8.1) and cuDNN (e.g., 9.15.1) <br> 2) Integrate Python 3.12 and commonly used system toolchains <br>3) Support proxy configuration and cache cleanup optimization|

**Technical Stack Details:**

| Component | Version Information | Role Description |
|-------------|-----------------|----------------|
| sccache | 0.12.0	| Use compilation cache to accelerate compilation |
| OpenMPI | 5.0.9	| 1. Multi-machine and multi-card collaboration to accelerate large-scale model parallel training<br>2. Automatically trigger integrated testing across multiple containers to verify the fault tolerance and performance of distributed systems<br>3. Ensure consistent MPI version and compilation parameters between local developer environments and production environments
| miniconda | latest	| 1. Quickly set up development environments for libraries like PyTorch <br>2. Build reproducible test environments in pipelines <br>3. Each service runs independently in its own Conda environment, reducing coupling |
| cuDNN | 9.15.1 | 1. Utilize GPU to accelerate scientific computing tasks <br>2. Deploy models accelerated by CuDNN in containers (e.g., Transformer) <br> 3. Run lightweight AI inference services on edge devices supporting NVIDIA GPUs


**Usage:**
```sh
# Build the FlagScale base image
# Supports CUDA 12.8.1, cuDNN 9.15.1, Python 3.12
docker build \
   --build-arg http_proxy=http://x.x.x.x:xxxx \
   --build-arg https_proxy=http://x.x.x.x:xxxx \
   --build-arg CUDA_VERSION=12.8.1 \
   -f docker/base/Dockerfile.cuda.base \
   -t flagscale:cuda12.8.1-cudnn9.15.1-python3.12-base .
```

## 2. Inference Service Image
**Customized images are provided according to different inference framework requirements:**

| Chip Type | Supported Backend | Configuration file |  Applicable Scenarios |
|---------|----------|---------|----------|
| Nvidia  | VLLM | docker/inference/Dockerfile.cuda.vllm | Build an inference image for Nvidia based on VLLM
| MetaX   | VLLM | docker/inference/Dockerfile.metax.vllm | Build an inference image for Metax based on VLLM


**Technical Stack Details:**
| Component | Version Information | Role Description |
|------|----------|--------|
| VLLM | unpath sync | Core framework of LLM inference engine |
|PyTorch | 2.7.1+cu128 | Basic runtime environment for deep learning, integrating CUDA 12.8 compiler and cuDNN 9.15.1 acceleration library |
| FlagGems | 3.0 | Self-developed operator library |

**Usage:**
```sh
# Build the FlagScale inference image
docker build \
   # Set HTTP/HTTPS proxy server address and port number
   --build-arg http_proxy=http://x.x.x.x:xxxx \
   --build-arg https_proxy=http://x.x.x.x:xxxx \
   # Base image name and version information
   --build-arg BASE_IMAGE=flagscale:cuda12.8.1-cudnn9.15.1-python3.12-base \
   # Specify the repository URL of the FlagScale project
   --build-arg FLAGSCALE_REPO=https://github.com/your-repo/FlagScale.git \
   # Select the branch to clone
   --build-arg FLAGSCALE_BRANCH=your-branch \
   # Select the commit to switch to
   --build-arg FLAGSCALE_COMMIT=your-commit \
   # PyTorch version number (includes CUDA 12.8 support)
   --build-arg PYTORCH_VER="2.7.1+cu128" \
   # TorchAudio version number (compatible with PyTorch)
   --build-arg TORCHAUDIO_VER="2.7.1+cu128" \
   # TorchVision version number (used for image processing tasks)
   --build-arg TORCHVISION_VER="0.22.1+cu128" \
   # Additional PyPI index addresses for installing specific versions of dependencies
   --build-arg EXTRA_INDEX="https://download.pytorch.org/whl/cu128" \
   # Specify the path of the Dockerfile to use
   -f docker/inference/Dockerfile.cuda.vllm \
   # Target image tag naming rule: <CUDA version>-<cuDNN version>-<Python version>-<PyTorch version>-<timestamp>-<purpose>
   -t flagscale:cuda12.8.1-cudnn9.15.1-python3.12-torch2.7.1-time2512031818-inference .

# Run the container
docker run -itd --gpus all --shm-size=500g \
   --name Inference2512031818 \
   --hostname flagscale_inference \
   -v /data/flagscale_cicd/docker/docker_build/docker_data:/home/gitlab-runner/data \
   -v /data/flagscale_cicd/docker/docker_build/docker_tokenizers:/home/gitlab-runner/tokenizers flagscale:cuda12.8.1-cudnn9.15.1-python3.12-torch2.7.1-time2512031818-inference

# Enter the container
docker exec -it Inference2512031818 bash

```

## 3. Training Task Image
Designed for large-scale distributed training:

| Chip Type | Supported Backend | Configuration File |  Applicable Scenarios |
|---------|----------|---------|----------|
| Nvidia  | megatron | docker/train/Dockerfile.cuda.megatron | Build a training image for Nvidia based on Megatron |

**Technical Stack Details:**
| Component | Version Information | Role Description |
|------|----------|--------|
| megatron | unpath sync | Large-scale distributed training framework supporting trillion-parameter model parallelism |
|PyTorch | 2.7.1+cu128 | Basic runtime environment for deep learning, integrating CUDA 12.8 compiler and cuDNN 9.15.1 acceleration library |
| TransformerEngine | e9a5fa4e | Efficient transformer computation engine with mixed-precision training and dynamic tensor parallelism support |
| Flash Attention | 2.8.0.post2 | Memory-optimized attention mechanism reducing memory usage via chunked computation |
| GROUPED GEMM | 1.1.4.post6 | Group general matrix multiplication library designed for sparse/structured tensors to improve MoE model inference throughput |
| APEX |  0.1 | NVIDIA official mixed-precision training toolchain supporting automatic loss scaling and gradient clipping |

**Usage**
```sh
# Build the FlagScale training image
# Supports CUDA 12.8.1, cuDNN 9.15.1, Python 3.12 and PyTorch 2.7.1+cu128
docker build \
   # Set HTTP/HTTPS proxy server address and port number
   --build-arg http_proxy=http://x.x.x.x:xxxx \
   --build-arg https_proxy=http://x.x.x.x:xxxx \
   # Base image name and version information
   --build-arg BASE_IMAGE=flagscale:cuda12.8.1-cudnn9.15.1-python3.12-base \
   # Specify the repository URL of the FlagScale project
   --build-arg FLAGSCALE_REPO=https://github.com/your-repo/FlagScale.git \
   # Select the branch to clone
   --build-arg FLAGSCALE_BRANCH=your-branch \
   # Select the commit to switch to
   --build-arg FLAGSCALE_COMMIT=your-commit \
   # PyTorch version number (includes CUDA 12.8 support)
   --build-arg PYTORCH_VER="2.7.1+cu128" \
   # TorchAudio version number (compatible with PyTorch)
   --build-arg TORCHAUDIO_VER="2.7.1+cu128" \
   # TorchVision version number (used for image processing tasks)
   --build-arg TORCHVISION_VER="0.22.1+cu128" \
   # Additional PyPI index addresses for installing specific versions of dependencies
   --build-arg EXTRA_INDEX="https://download.pytorch.org/whl/cu128" \
   # Transformer Engine commit hash (refer to official documentation to ensure compatibility with the selected PyTorch version)
   --build-arg TRANSFORMER_ENGINE_COMMIT="e9a5fa4e" \
   # Flash Attention library version number (refer to official documentation to ensure compatibility with the selected PyTorch version)
   --build-arg FLASH_ATTN_VERSION="2.8.0.post2" \
   # Grouped Gemmi library version number (refer to official documentation to ensure compatibility with the selected PyTorch version)
   --build-arg GROUPED_GEMM_VERSION="1.1.4.post6" \
   # Specify the path of the Dockerfile to use
   -f docker/train/Dockerfile.cuda.megatron \
   # Target image tag naming rule: <CUDA version>-<cuDNN version>-<Python version>-<PyTorch version>-<timestamp>-<purpose>
   -t flagscale:cuda12.8.1-cudnn9.15.1-python3.12-torch2.7.1-time2512031818-train .

# Run the container
docker run -itd --gpus all --shm-size=500g \
   --name Train2512031818 \
   --hostname flagscale_train \
   -v /data/flagscale_cicd/docker/docker_build/docker_data:/home/gitlab-runner/data \
   -v /data/flagscale_cicd/docker/docker_build/docker_tokenizers:/home/gitlab-runner/tokenizers flagscale:cuda12.8.1-cudnn9.15.1-python3.12-torch2.7.1-time2512031818-train

# Enter the container
docker exec -it Train2512031818 bash

```

## 4. Reinforcement Learning (RL) Image
Dedicated environment for RL algorithm development:

- **Configuration File:** docker/rl/Dockerfile.cuda.verl
- **Key Components:**
   - Pre-installed Verl framework (including Flash Attention dependencies)
   - Dynamic resource scheduling support
   - Compatibility with GroupedGEMM operator library

**Usage**
```sh
# Build the FlagScale reinforcement learning image
# Supports CUDA 12.8.1, cuDNN 9.15.1, Python 3.10 and PyTorch
docker build \
   # Set HTTP/HTTPS proxy server address and port number
   --build-arg http_proxy=http://x.x.x.x:xxxx \
   --build-arg https_proxy=http://x.x.x.x:xxxx \
   # Base image name and version information
   --build-arg BASE_IMAGE=flagscale:cuda12.8.1-cudnn9.15.1-python3.12-base \
   # Specify the repository URL of the FlagScale project
   --build-arg FLAGSCALE_REPO=https://github.com/your-repo/FlagScale.git \
   # Select the branch to clone
   --build-arg FLAGSCALE_BRANCH=your-branch \
   # Select the commit to switch to
   --build-arg FLAGSCALE_COMMIT=your-commit \
   # Specify the path of the Dockerfile to use
   -f docker/rl/Dockerfile.cuda.verl \
   # Target image tag naming rule: <CUDA version>-<cuDNN version>-<Python version>-<PyTorch version>-<timestamp>-<purpose>
   -t flagscale:cuda12.8.1-cudnn9.15.1-python3.10-torch2.6.0-time2512031818-rl .

# Run the container
docker run -itd --gpus all --shm-size=500g \
   --name RL2512031818 \
   --hostname flagscale_train \
   -v /data/flagscale_cicd/docker/docker_build/docker_data:/home/gitlab-runner/data \
   -v /data/flagscale_cicd/docker/docker_build/docker_tokenizers:/home/gitlab-runner/tokenizers flagscale:cuda12.8.1-cudnn9.15.1-python3.10-torch2.6.0-time2512031818-rl

# Enter the container
docker exec -it RL2512031818 bash
```

## 5. Full-Featured Integrated Image
One-click access to a complete development environment:

- Configuration File: docker/Dockerfile.cuda
- Advantages:
   - Aggregates inference/training/RL environments

**Usage:**
```sh
docker build \
   # Set HTTP/HTTPS proxy server address and port number
   --build-arg http_proxy=http://x.x.x.x:xxxx \
   --build-arg https_proxy=http://x.x.x.x:xxxx \
   # Base image name and version information
   --build-arg FLAGSCALE_BASE=flagscale:cuda12.8.1-cudnn9.15.1-python3.12-base \
   # Inference image name and version information
   --build-arg FLAGSCALE_INFERENCE=flagscale:cuda12.8.1-cudnn9.15.1-python3.12-torch2.7.1-time2512031818-inference \
   # Training image name and version information
   --build-arg FLAGSCALE_TRAIN=flagscale:cuda12.8.1-cudnn9.15.1-python3.12-torch2.7.1-time2512031818-train \
   # Reinforcement learning image name and version information
   --build-arg FLAGSCALE_RL=flagscale:cuda12.8.1-cudnn9.15.1-python3.10-torch2.6.0-time2512031818-rl \
   # Specify the path of the Dockerfile to use
   -f docker/Dockerfile.cuda \
   # Target image tag naming rule: <CUDA version>-<cuDNN version>-<Python version>-<PyTorch version>-<timestamp>
   -t flagscale:cuda12.8.1-cudnn9.15.1-python3.10-torch2.6.0-time2512031818 .
```

## 6. SSH Passwordless Login Enhancement
**Security Warning & Operational Advice**

- This image construction method has security risks and is only intended for internal development environments. Do not distribute built image files externally.
- To avoid environmental contamination caused by identical keys, it is strongly recommended to rebuild new images for each task.

```sh
docker build \
   --build-arg BASE_IMAGE=flagscale:cuda12.4.1-cudnn9.5.0-python3.12-torch2.5.1-time2503251131 
   --build-arg SSH_PORT=22 
   -f docker/Dockerfile.ssh 
   -t flagscale:cuda12.4.1-cudnn9.5.0-python3.12-torch2.5.1-time2503251131-ssh .
```
