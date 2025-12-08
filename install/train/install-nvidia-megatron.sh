#!/bin/bash

set -e

print_help() {
cat << EOF
Usage: $0 [OPTIONS]

Options:
  --env <inference>                  Specify the environment type (required)
  --torch-ver <version>              Specify the PyTorch version (e.g., "2.7.1+cu128")
  --torchaudio-ver <version>         Specify the TorchAudio version (e.g., "2.7.1+cu128")
  --torchvision-ver <version>        Specify the TorchVision version (e.g., "0.22.1+cu128")
  --extra_index <url>                Specify an extra index URL for pip (optional)
  --flash-attn-ver <version>         Specify the Flash Attention version (e.g., "2.8.0.post2")
  --group-gemm-ver <version>         Specify the Grouped GEMM version (e.g., "1.1.4.post6")
  --transformer-engine-commit <commit> Specify the Transformer Engine commit hash (e.g., "e9a5fa4e")
  -h|--help                          Show this help message and exit

Example:
  $0 --env "inference" \
      --torch-ver "2.7.1+cu128" --torchaudio-ver "2.7.1+cu128" --torchvision-ver "0.22.1+cu128" --extra-index "https://download.pytorch.org/whl/cu128" \
      --flash-attn-ver "2.8.0.post2" --group-gemm-ver "1.1.4.post6" --transformer-engine-commit "e9a5fa4e"
EOF
}

# Define variables for clarity and maintainability
# Initialize the variable
env=""
PYTORCH_VER="2.7.1+cu128"
TORCHAUDIO_VER="2.7.1+cu128"
TORCHVISION_VER="0.22.1+cu128"
EXTRA_INDEX="https://download.pytorch.org/whl/cu128"
FLASH_ATTN_VERSION="2.8.0.post2"
GROUPED_GEMM_VERSION="1.1.4.post6"
TRANSFORMER_ENGINE_COMMIT="e9a5fa4e"

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) env="$2"; shift ;;  # Assign the value after '--env'
        --torch-ver) PYTORCH_VER="$2"; shift ;;
        --torchaudio-ver) TORCHAUDIO_VER="$2"; shift ;;
        --torchvision-ver) TORCHVISION_VER="$2"; shift ;;
        --extra_index) EXTRA_INDEX="$2"; shift ;;
        --flash-attn-ver) FLASH_ATTN_VERSION="$2"; shift ;;
        --group-gemm-ver) GROUPED_GEMM_VERSION="$2"; shift ;;
        --transformer-engine-commit) TRANSFORMER_ENGINE_COMMIT="$2"; shift ;;
        --help|-h) print_help; exit 0 ;;
        *) echo "Error: Unknown parameter passed."; print_help; exit 1 ;;
    esac
    shift
done

# Validate that the 'env' parameter is provided and must be set to 'train'.
if [ -z "$env" ]; then
    echo "Error: env field is required.."
    exit 1
fi

# Check the value of env
if [ "$env" != "train" ]; then
    echo "Error: 'env' must be 'train'. Current value: ${env}"
    exit 1
fi

# Load the installation script to set up the NVIDIA-related environment (including PyTorch and its components).
source ./install/common/install-nvidia-common.sh \
    --torch-ver ${PYTORCH_VER} \
    --torchaudio-ver ${TORCHAUDIO_VER} \
    --torchvision-ver ${TORCHVISION_VER} \
    --extra-index ${EXTRA_INDEX} \
    --flash-attn-ver ${FLASH_ATTN_VERSION} \
    --group-gemm-ver ${GROUPED_GEMM_VERSION} \
    --transformer-engine-commit ${TRANSFORMER_ENGINE_COMMIT}

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh

# Accept the TOS from the main channel
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# Accept the TOS from the r channel
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create and activate Conda virtual environment
# The Python version used has been written into the conda config
if conda env list | grep -q "flagscale-${env}"; then
    # Check if the environment already exists
    echo "Conda environment 'flagscale-${env}' already exists."
else
    echo "Creating conda environment 'flagscale-${env}'..."
    # Use the current Python version of the system (e.g. 3.12)
    conda create --name "flagscale-${env}" python=$(python --version | awk '{print $2}' | cut -d '.' -f 1,2) -y
fi

# Activate the target Conda environment
conda activate flagscale-${env}

# Install torch
install_torch

# Install transformer_engine
install_transformer_engine

# Verify cuDNN
python -c "import torch; print('cuDNN version:', torch.backends.cudnn.version());"

# Install flash-attn
install_flash_attn

# Install grouped_gemm
install_grouped_gemm

# apex train
install_apex

# Used for automatic fault tolerance
# Set the path to the target Python file
SITE_PACKAGES_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])")
FILE="$SITE_PACKAGES_DIR/torch/distributed/elastic/agent/server/api.py"
torch_version=`python -c "import torch; print(torch.__version__)"`
echo "torch_version: $torch_version"

# Replace the following code with torch version 2.5.1
if [[ $torch_version == *"2.5.1"* ]];then
    # Check and replace line 893
    LINE_893=$(sed -n '893p' "$FILE")
    EXPECTED_893='                if num_nodes_waiting > 0:'

    if [[ "$LINE_893" != "$EXPECTED_893" ]]; then
        echo "Error: Line 893 in $FILE does not exactly match '                if num_nodes_waiting > 0:' ."
        exit 1
    else
        echo "Line 893 is correct. Proceeding with replacement."
        # Directly replace the line without using regex
        sed -i '893s|.*|                if num_nodes_waiting > 0 and self._remaining_restarts > 0:|' "$FILE"
        echo "Success: Line 893 replaced."
    fi

    # Check and replace line 902
    LINE_902=$(sed -n '902p' "$FILE")
    EXPECTED_902='                    self._restart_workers(self._worker_group)'

    if [[ "$LINE_902" != "$EXPECTED_902" ]]; then
        echo "Error: Line 902 does not match '                    self._restart_workers(self._worker_group)'."
        exit 1
    else
        echo "Line 902 is correct. Proceeding with replacement."
        # Directly replace the line without using regex
        sed -i '902s|.*|                    self._remaining_restarts -= 1; self._restart_workers(self._worker_group)|' "$FILE"
        echo "Success: Line 902 replaced."
    fi
fi

# Replace the following code with torch version 2.6.0
if [[ $torch_version == *"2.6.0"* ]] || [[ $torch_version == *"2.7.0"* ]]  || [[ $torch_version == *"2.7.1"* ]];then
    # Check and replace line 908
    LINE_908=$(sed -n '908p' "$FILE")
    EXPECTED_908='                if num_nodes_waiting > 0:'

    if [[ "$LINE_908" != "$EXPECTED_908" ]]; then
        echo "Error: Line 908 in $FILE does not exactly match '                if num_nodes_waiting > 0:'."
        exit 1
    else
        echo "Line 908 is correct. Proceeding with replacement."
        # Directly replace the line without using regex
        sed -i '908s|.*|                if num_nodes_waiting > 0 and self._remaining_restarts > 0:|' "$FILE"
        echo "Success: Line 908 replaced."
    fi

    # Check and replace line 917
    LINE_917=$(sed -n '917p' "$FILE")
    EXPECTED_917='                    self._restart_workers(self._worker_group)'

    if [[ "$LINE_917" != "$EXPECTED_917" ]]; then
        echo "Error: Line 917 does not match '                    self._restart_workers(self._worker_group)'."
        exit 1
    else
        echo "Line 917 is correct. Proceeding with replacement."
        # Directly replace the line without using regex
        sed -i '917s|.*|                    self._remaining_restarts -= 1; self._restart_workers(self._worker_group)|' "$FILE"
        echo "Success: Line 917 replaced."
    fi
fi
