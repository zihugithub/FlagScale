#!/bin/bash

set -e

print_help() {
cat << EOF
Usage: $0 [OPTIONS]

Options:
  --env <inference>              Specify the environment type (required)
  --torch-ver <version>          Specify the PyTorch version (e.g., "2.6.0+metax3.0.0.3")
  --torchaudio-ver <version>     Specify the TorchAudio version (e.g., "2.4.1+metax3.0.0.3")
  --torchvision-ver <version>    Specify the TorchVision version (e.g., "0.15.1+metax3.0.0.3")
  -h|--help                      Show this help message and exit

Example:
  $0 --env "inference" \
    --torch-ver "2.6.0+metax3.0.0.3" \
    --torchaudio-ver "2.4.1+metax3.0.0.3" 
    --torchvision-ver "0.15.1+metax3.0.0.3" 
EOF
}

# Initialize the variable
env=""
PYTORCH_VER="2.6.0+metax3.0.0.3"
TORCHAUDIO_VER="2.4.1+metax3.0.0.3"
TORCHVISION_VER="0.15.1+metax3.0.0.3"

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) env="$2"; shift ;;
        --torch-ver) PYTORCH_VER="$2"; shift ;;
        --torchaudio-ver) TORCHAUDIO_VER="$2"; shift ;;
        --torchvision-ver) TORCHVISION_VER="$2"; shift ;;
        --help|-h) print_help; exit 0 ;;
        *) echo "Error: Unknown parameter passed."; print_help; exit 1 ;;
    esac
    shift
done

# Validate that the 'env' parameter is provided and must be set to 'inference'.
if [ -z "$env" ]; then
    echo "Error: env field is required.."
    exit 1
fi

# Check the value of env
if [ "$env" != "inference" ]; then
    echo "Error: 'env' must be 'inference'. Current value: ${env}"
    exit 1
fi

# Load the installation script to set up the METAX-related environment (including PyTorch and its components).
source ./install/common/install-metax-common.sh \
    --torch-ver ${PYTORCH_VER} \
    --torchaudio-ver ${TORCHAUDIO_VER} \
    --torchvision-ver ${TORCHVISION_VER}

# Proceed with setup based on the value of 'env'
echo "Setting up environment for: $env"

# Load conda environment
source /etc/profile.d/conda.sh

# Create and activate Conda virtual environment
# The Python version used has been written into the conda config
if conda env list | grep -q "flagscale-${env}"; then
    # Check if the environment already exists
    echo "Conda environment 'flagscale-${env}' already exists."
else
    echo "Creating conda environment 'flagscale-${env}'..."
    # Create an flagscale-${env} environment based on the base environment
    conda create --name "flagscale-${env}" --clone base
fi

# Activate the target Conda environment
conda activate flagscale-${env}

# install torch
install_torch

echo "[INFO] Entering inference mode setup..."
# Basic dependency installation
if ! command -v pip &> /dev/null; then
    echo "Error: pip not found. Please install Python package manager first."
    exit 1
fi

# Perform unpath operation
echo "python tools/patch/unpatch.py --backend vllm FlagScale --task inference --device-type Metax_C550 ..."
python tools/patch/unpatch.py \
    --backend vllm \
    FlagScale \
    --task inference \
    --device-type Metax_C550 || {
    echo "Failed to execute unpatch script"; exit 1
}

# Preparation for Building Environment
build_dir="build/Metax_C550/FlagScale/third_party/vllm"
if [ ! -d "$build_dir" ]; then
    echo "Build directory not found: $build_dir"; exit 1
fi

# Enter the target directory and record the original path
pushd "$build_dir" > /dev/null

WHEEL_PATH="dist/vllm-*-linux_x86_64.whl"

{
    # Load environment variables
    source ./env.sh || { echo "Failed to source environment files"; exit 1; }

echo "[INFO] Checking sccache installation ..."
    if ! command -v sccache &> /dev/null; then
        echo "[INFO] sccache not found, installing..."
        curl -L https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-x86_64-unknown-linux-musl.tar.gz -o /tmp/sccache.tar.gz
        tar -xzf /tmp/sccache.tar.gz -C /tmp
        mv /tmp/sccache-v0.8.1-x86_64-unknown-linux-musl/sccache /usr/bin/sccache
        rm -rf /tmp/sccache*
    fi
    sccache --version

    echo "[INFO] Configuring compiler cache (sccache) ..."
    export CC=/root/cu-bridge/bin/gcc
    export CXX=/root/cu-bridge/bin/g++
    export CMAKE_C_COMPILER_LAUNCHER=$(which sccache)
    export CMAKE_CXX_COMPILER_LAUNCHER=$(which sccache)
    export CMAKE_CUDA_COMPILER_LAUNCHER=$(which sccache)
    export CXXFLAGS="-I/opt/maca/include -I/opt/maca/include/mcr -I/opt/maca/include/common -I/opt/maca/include/mcsparse -I/opt/maca/include/mcsolver"

    sccache --start-server || true

    echo "Setting up Python environment ..."
    python setup.py bdist_wheel || { echo "Failed to build wheel"; exit 1; }

    echo "Installing custom wheel package..."
    pip uninstall -y vllm
    pip install -v "$WHEEL_PATH" || { echo "Failed to install wheel"; exit 1; }

    echo "[INFO] sccache build statistics:"
    sccache --show-stats || true
} || {
    echo "Build process failed in $(pwd)"
    popd > /dev/null
    exit 1
}
popd > /dev/null

echo 'Inference environment setup completed successfully.'

# Clean all conda caches
conda clean --all -y
