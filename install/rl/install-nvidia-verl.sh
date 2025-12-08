#!/bin/bash

set -e

print_help() {
cat << EOF
Usage: $0 [OPTIONS]

Options:
  --env <RL>                  Specify the environment type (required)
  -h|--help                          Show this help message and exit

Example:
  $0 --env "RL"
EOF
}

# Define variables for clarity and maintainability
# Initialize the variable
env=""

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) env="$2"; shift ;;  # Assign the value after '--env'
        --help|-h) print_help; exit 0 ;;
        *) echo "Error: Unknown parameter passed."; print_help; exit 1 ;;
    esac
    shift
done

# Validate that the 'env' parameter is provided and must be set to 'RL'.
if [ -z "$env" ]; then
    echo "Error: env field is required.."
    exit 1
fi

# Check the value of env
if [ "$env" != "RL" ]; then
    echo "Error: 'env' must be 'RL'. Current value: ${env}"
    exit 1
fi

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
    # Use the current Python version of the system (e.g. 3.10)
    conda create --name "flagscale-${env}" python==3.10 -y
fi

# Activate the target Conda environment
conda activate flagscale-${env}

python -m pip install --upgrade pip

# This command updates `setuptools` to the latest version, ensuring compatibility and access to the latest features for Python package management.
pip install --upgrade setuptools


# install verl
python tools/patch/unpatch.py --backend verl
cd third_party/verl
# Install verl
if command -v uv &> /dev/null; then
    echo "Using 'uv' for installation..."
    uv pip install --no-deps -e .
else
    echo "Using 'pip' for installation..."
    pip install --no-deps -e .
fi
bash scripts/install_vllm_sglang_mcore.sh

# Install dependencies
if command -v uv &> /dev/null; then
    echo "Using 'uv' for installation..."
    uv pip install cryptography
else
    echo "Using 'pip' for installation..."
    pip install cryptography
fi
