#!/bin/bash

set -e

print_help() {
    echo "Usage: $0 [--env <train|inference>] [--llama-cpp-backend <cpu|metal|blas|openblas|blis|cuda|gpu|musa|vulkan_mingw64|vulkan_msys2|cann|arm_kleidi|hip|opencl_android|opencl_windows_arm64>]"
    echo "Options:"
    echo "  --env <train|inference>         Specify the environment type (required)"
    echo "  --llama-cpp-backend <backend>   Specify the llama.cpp backend (default: cpu)"
}

# Initialize the variable
env=""
llama_cpp_backend="cpu"

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) env="$2"; shift ;;  # Assign the value after '--env'
        --llama-cpp-backend) llama_cpp_backend="$2"; shift ;;  # Assign the value after '--llama-cpp-backend'
        --help|-h) print_help; exit 0 ;;
        *) echo "Error: Unknown parameter passed."; print_help; exit 1 ;;
    esac
    shift
done

# Check if 'env' field is provided and is either 'train' or 'inference'
if [ -z "$env" ]; then
    echo "Error: env field is required. Please specify either 'train' or 'inference'."
    exit 1
fi

# Check the value of env
if [ "$env" != "train" ] && [ "$env" != "inference" ]; then
    echo "Error: env must be 'train' or 'inference'."
    exit 1
fi

python -m pip install --upgrade pip

# Packages that need to be installed outside of the conda environment
pip install -r ./requirements/requirements-base.txt

# Proceed with setup based on the value of 'env'
echo "Setting up environment for: $env"

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh

# Create and activate Conda virtual environment
# The Python version used has been written into the conda config
if conda env list | grep -q "flagscale-${env}"; then
    # Check if the environment already exists
    echo "Conda environment 'flagscale-${env}' already exists."
else
    echo "Creating conda environment 'flagscale-${env}'..."
    conda create --name "flagscale-${env}" python=$(python --version | awk '{print $2}' | cut -d '.' -f 1,2) -y
fi
conda activate flagscale-${env}


wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
rm flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# From Megatron-LM log
pip install "git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2#egg=flashattn-hopper&subdirectory=hopper"
python_path=`python -c "import site; print(site.getsitepackages()[0])"`
mkdir -p $python_path/flashattn_hopper
wget -P $python_path/flashattn_hopper https://raw.githubusercontent.com/Dao-AILab/flash-attention/v2.7.2/hopper/flash_attn_interface.py

# If env equals 'inference'
if [ "${env}" == "inference" ]; then
    # Unpatch
    python tools/patch/unpatch.py --backend vllm
    python tools/patch/unpatch.py --backend llama.cpp

    # Build vllm
    # Navigate to requirements directory and install inference dependencies
    pip install -r ./third_party/vllm/requirements/build.txt
    pip install -r ./third_party/vllm/requirements/cuda.txt
    pip install -r ./third_party/vllm/requirements/common.txt
    pip install "git+https://github.com/state-spaces/mamba.git@v2.2.4"
    pip install -r ./third_party/vllm/requirements/dev.txt
    
    MAX_JOBS=$(nproc) pip install --no-build-isolation -v ./third_party/vllm/.

    # Navigate to requirements directory and install serving dependencies
    pip install -r ./requirements/serving/requirements.txt

    # Build llama.cpp
    cd ./third_party/llama.cpp
    rm -rf ./build
    case "$llama_cpp_backend" in
        cpu|metal|cpu_and_metal)
            cmake -B build
            cmake --build build --config Release -j8
            ;;
        blas|openblas)
            cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
            cmake --build build --config Release
            ;;
        blis)
            # You can skip this step if  in oneapi-basekit docker image, only required for manual installation
            source /opt/intel/oneapi/setvars.sh
            cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_NATIVE=ON
            cmake --build build --config Release
            ;;
        cuda|gpu)
            cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
            cmake --build build --config Release
            ;;
        musa)
            cmake -B build -DGGML_MUSA=ON
            cmake --build build --config Release
            ;;
        vulkan_mingw64)
            cmake -B build -DGGML_VULKAN=ON
            cmake --build build --config Release
            ;;
        cann)
            cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release
            cmake --build build --config release
            ;;
        arm_kleidi)
            cmake -B build -DGGML_CPU_KLEIDIAI=ON
            cmake --build build --config Release
            ;;
        hip|vulkan_w64devkit|vulkan_msys2|opencl_android|opencl_windows_arm64)
            echo "auto build unsupport: $1, follow the README.md to build manually:"
            echo "https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md"
            exit 1
            ;;  
        *)
            echo "unknown backend: $1"
            print_help
            exit 1
            ;;
    esac

    # For FlagRelease
    pip install --no-build-isolation git+https://github.com/FlagOpen/FlagGems.git@release_v1.0.0
fi

# Clean all conda caches
conda clean --all -y
