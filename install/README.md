# Environment Installation Guide
This document explains how to use Conda's isolation features to build environments for different purposes. Please ensure Miniconda is installed before proceeding with the following operations.

## 1. Install Miniconda (if not already installed)
```sh
# Create installation directory and download latest Miniconda
mkdir -p ~/miniconda3 && \
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \

# Silent installation and cleanup
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
rm -rf ~/miniconda3/miniconda.sh && \

# Initialize Conda and configure default settings
~/miniconda3/bin/conda init bash && \
~/miniconda3/bin/conda config --set auto_activate_base false && \
~/miniconda3/bin/conda config --set default_python 3.12
```

**Tip:** After installation, it's recommended to restart your terminal or run source ~/miniconda3/etc/profile.d/conda.sh to activate Conda.

# 2. Training Environment Installation (Megatron)

```bash
bash ./install/train/install-nvidia-megatron.sh \
    --env train \                                              # Required: specify environment name
    --torch-ver "2.7.1+cu128" \                                # PyTorch version
    --torchaudio-ver "2.7.1+cu128" \                           # TorchAudio version
    --torchvision-ver "0.22.1+cu128" \                         # TorchVision version
    --extra-index "https://download.pytorch.org/whl/cu128" \   # Custom PyTorch repository
    --flash-attn-ver "2.8.0.post2" \                           # Flash Attention version
    --group-gemm-ver "1.1.4.post6" \                           # Group GEMM version
    --transformer-engine-commit "e9a5fa4e"                     # Transformer Engine commit ID

```

# 3. Inference Environment Installation (vLLM)

```bash
bash ./install/inference/install-nvidia-vllm.sh \
    --env inference \                                          # Required: specify environment name
    --llama-cpp-backend "cpu" \                                # LlamaCPP backend
    --omni_infer "0" \                                         # Omni Inference switch
    --torch-ver "2.7.1+cu128" \                                # PyTorch version
    --torchaudio-ver "2.7.1+cu128" \                           # TorchAudio version
    --torchvision-ver "0.22.1+cu128" \                         # TorchVision version
    --extra-index "https://download.pytorch.org/whl/cu128"     # Custom PyTorch repository

```

# 4. Reinforcement Learning Environment Installation (Verl)

```bash
bash ./install/rl/install-nvidia-verl.sh \
    --env RL                                                    # Required: specify environment name
```

# 5.  MetaX Inference Environment Installation

```bash
bash ./install/inference/install-metax-vllm.sh \
    --env "inference" \                                         # Required: specify environment name
    --torch-ver "2.6.0+metax3.0.0.3" \                          # MetaX compatible PyTorch
    --torchaudio-ver "2.4.1+metax3.0.0.3" \                     # MetaX compatible TorchAudio
    --torchvision-ver "0.15.1+metax3.0.0.3"                     # MetaX compatible TorchVision
```
