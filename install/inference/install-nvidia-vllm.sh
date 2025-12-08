#!/bin/bash

set -e

print_help() {
cat << EOF
Usage: $0 [OPTIONS]

Options:
  --env <inference>                  Specify the environment type (required)
  --llama-cpp-backend <backend>      Specify the llama.cpp backend (default: cpu)
  --omni_infer <value>               Specify omni inference value (optional)
  --torch-ver <version>              Specify the PyTorch version (e.g., "2.7.1+cu128")
  --torchaudio-ver <version>         Specify the TorchAudio version (e.g., "2.7.1+cu128")
  --torchvision-ver <version>        Specify the TorchVision version (e.g., "0.22.1+cu128")
  --extra_index <url>                Specify an extra index URL for pip (optional)
  -h|--help                          Show this help message and exit

Example:
  $0 --env "inference" --llama-cpp-backend "cpu"  --omni_infer "0" \
      --torch-ver "2.7.1+cu128" --torchaudio-ver "2.7.1+cu128" --torchvision-ver "0.22.1+cu128" --extra-index "https://download.pytorch.org/whl/cu128"
EOF
}

# Define variables for clarity and maintainability
# Initialize the variable
env=""
llama_cpp_backend="cpu"
omni_infer="0"
PYTORCH_VER="2.7.1+cu128"
TORCHAUDIO_VER="2.7.1+cu128"
TORCHVISION_VER="0.22.1+cu128"
EXTRA_INDEX="https://download.pytorch.org/whl/cu128"

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) env="$2"; shift ;;  # Assign the value after '--env'
        --llama-cpp-backend) llama_cpp_backend="$2"; shift ;;
        --omni_infer) omni_infer="$2"; shift ;;
        --torch-ver) PYTORCH_VER="$2"; shift ;;
        --torchaudio-ver) TORCHAUDIO_VER="$2"; shift ;;
        --torchvision-ver) TORCHVISION_VER="$2"; shift ;;
        --extra_index) EXTRA_INDEX="$2"; shift ;;
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

# Load the installation script to set up the NVIDIA-related environment (including PyTorch and its components).
source ./install/common/install-nvidia-common.sh \
    --torch-ver ${PYTORCH_VER} \
    --torchaudio-ver ${TORCHAUDIO_VER} \
    --torchvision-ver ${TORCHVISION_VER} \
    --extra-index ${EXTRA_INDEX}

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

# install torch
install_torch

# Verify cuDNN
python -c "import torch; print('cuDNN version:', torch.backends.cudnn.version());"

# Unpatch
python tools/patch/unpatch.py --backend vllm
python tools/patch/unpatch.py --backend llama.cpp
python tools/patch/unpatch.py --backend omniinfer

# Build vllm
# Navigate to requirements directory and install inference dependencies
echo $SCCACHE_DIR
which sccache
sccache --version
sccache --start-server
sccache --show-stats
install_or_verify_vllm
sccache --show-stats

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

cd ../..
# Build omniinfer
if [ "${omni_infer}" == "1" ]; then
    # process repo
    find ./third_party/omniinfer -type f -exec dos2unix {} +
    find ./third_party/omniinfer -type f -path '*.sh' -exec chmod a+x {} \;

    # unpatch vllm
    cd ./third_party/omniinfer/infer_engines/
    git clone https://github.com/vllm-project/vllm.git
    git checkout 65334ef3
    bash bash_install_code.sh
    cd ../../..

    # install dependencies
    if command -v uv &> /dev/null; then
        uv pip install -r ./third_party/omniinfer/tests/requirements.txt
    else
        pip install -r ./third_party/omniinfer/tests/requirements.txt
    fi

    # build whl for vllm
    mkdir -p ./third_party/omniinfer/build/dist
    cd ./third_party/omniinfer/infer_engines/vllm
    VLLM_TARGET_DEVICE=empty python setup.py bdist_wheel
    mv dist/vllm* ../../build/dist

    # build whl for omniinfer
    cd ../..
    pip install build
    python -m build
    mv dist/omni_i* ./build/dist

    # build whl for omniinfer omni_placement
    cd ./omni/accelerators/placement
    python setup.py bdist_wheel
    mv dist/omni_placement* ../../../build/dist

    # install 3 whl
    cd ../../../build/dist
    pip install omni_i*.whl
    pip install vllm*.whl
    pip install omni_placement*.whl

    cd ../../../..
fi

# For FlagRelease
# Installing grouped_gemm...
install_or_verify_flag_gems
