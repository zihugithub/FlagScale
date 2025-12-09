#!/bin/bash

print_help() {
cat << EOF
Usage: $0 [OPTIONS]

Options:
  --torch-ver <version>          Specify the PyTorch version (e.g., "2.7.1+cu128")
  --torchaudio-ver <version>     Specify the TorchAudio version (e.g., "2.7.1+cu128")
  --torchvision-ver <version>    Specify the TorchVision version (e.g., "0.22.1+cu128")
  --extra-index <url>            Specify an extra index URL for pip (optional)
  --flash-attn-ver <version>     Specify the Flash Attention version (e.g., "2.8.0.post2")
  --group-gemm-ver <version>     Specify the Grouped GEMM version (e.g., "1.1.4.post6")
  --transformer-engine-commit <commit> Specify the Transformer Engine commit hash (e.g., "e9a5fa4e")
  -h|--help                      Show this help message and exit

Example:
  $0 --torch-ver "2.7.1+cu128" --torchaudio-ver "2.7.1+cu128" --torchvision-ver "0.22.1+cu128" --extra-index "https://download.pytorch.org/whl/cu128" \
      --flash-attn-ver "2.8.0.post2" --group-gemm-ver "1.1.4.post6" --transformer-engine-commit "e9a5fa4e"
EOF
}

# Define variables for clarity and maintainability
# Initialize the variable
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
        --torch-ver) PYTORCH_VER="$2"; shift ;;
        --torchaudio-ver) TORCHAUDIO_VER="$2"; shift ;;
        --torchvision-ver) TORCHVISION_VER="$2"; shift ;;
        --extra-index) EXTRA_INDEX="$2"; shift ;;
        --flash-attn-ver) FLASH_ATTN_VERSION="$2"; shift ;;
        --group-gemm-ver) GROUPED_GEMM_VERSION="$2"; shift ;;
        --transformer-engine-commit) TRANSFORMER_ENGINE_COMMIT="$2"; shift ;;
        --help|-h) print_help; exit 0 ;;
        *) echo "Error: Unknown parameter passed."; print_help; exit 1 ;;
    esac
    shift
done

# Upgrade pip and setuptools.
python -m pip install --upgrade pip setuptools

# Checkout to get torch version (major.minor)
get_installed_version() {
    if pip show "$1" > /dev/null 2>&1; then
        pip show "$1" | grep "^Version:" | awk '{print $2}'
        return 0
    else
        return 1
    fi
}

# Checkout to get installed version of flash-attn
get_installed_flash_attn_version() {
    if pip show flash-attn > /dev/null 2>&1; then
        pip show flash-attn | grep "^Version:" | awk '{print $2}'
        return 0
    else
        return 1
    fi
}

# Checkout to get installed version of grouped_gemm
get_installed_grouped_gemm_version() {
    if pip show nv-grouped-gemm > /dev/null 2>&1; then
        pip show nv-grouped-gemm | grep "^Version:" | awk '{print $2}'
        return 0
    else
        return 1
    fi
}

# Checkout to get installed version of transformer_engine
get_installed_transformer_engine() {
    if pip show transformer_engine > /dev/null 2>&1; then
        pip show transformer_engine | grep "^Version:" | awk '{print $2}'
        return 0
    else
        return 1
    fi
}

# nstall or verify torch and its related libraries
install_torch() {
    local packages=("torch" "torchaudio" "torchvision")
    local installed_all_correct=true

    for package in "${packages[@]}"; do
        local expected_version
        case "$package" in
            "torch") expected_version="$PYTORCH_VER" ;;
            "torchaudio") expected_version="$TORCHAUDIO_VER" ;;
            "torchvision") expected_version="$TORCHVISION_VER" ;;
            *) expected_version="" ;;
        esac

        local installed_version
        installed_version=$(get_installed_version "$package")

        if [[ -n "$installed_version" ]]; then
            echo "Detected ${package} is already installed. Current version: ${installed_version}"
            if [[ "$installed_version" == "$expected_version" ]]; then
                echo "${package} version is correct."
            else
                echo "$ ${package} version does not match. Expected: ${expected_version}, Actual: ${installed_version}"
                installed_all_correct=false
            fi
        else
            echo "No installation detected for ${package}. Starting installation..."
            installed_all_correct=false
        fi
    done

    # If all packages are correctly installed, no further action needed
    if $installed_all_correct; then
        echo "All specified packages are already correctly installed."
        return 0
    fi

    # Proceed with installation
    echo "Starting installation or reinstallation of required packages..."
    if command -v uv &> /dev/null; then
        echo "Using 'uv' for installation..."
        uv pip install --break-system-packages \
            torch=="$PYTORCH_VER" \
            torchaudio=="$TORCHAUDIO_VER" \
            torchvision=="$TORCHVISION_VER" \
            --extra-index-url "$EXTRA_INDEX" || { echo "Installation failed. Please check network connection and permission settings."; exit 1; }
    else
        echo "Using 'pip' for installation..."
        pip install \
            torch=="$PYTORCH_VER" \
            torchaudio=="$TORCHAUDIO_VER" \
            torchvision=="$TORCHVISION_VER" \
            --extra-index-url "$EXTRA_INDEX" || { echo "Installation failed. Please check network connection and permission settings."; exit 1; }
    fi
}

# Install flash-attn
# Install Megatron-LM CUDA dependencies
install_flash_attn() {
    # Get cuda|torch|python|cxx versions
    cu=$(nvcc --version | grep "Cuda compilation tools" | awk '{print $5}' | cut -d '.' -f 1)
    torch=$(pip show torch | grep Version | awk '{print $2}' | cut -d '+' -f 1 | cut -d '.' -f 1,2)
    cp=$(python3 --version | awk '{print $2}' | awk -F. '{print $1$2}')
    cxx=$(g++ --version | grep 'g++' | awk '{print $3}' | cut -d '.' -f 1)
    flash_attn_version="${FLASH_ATTN_VERSION}"

    if [[ $(get_installed_flash_attn_version) == "${flash_attn_version}" ]]; then
        echo "flash-attn v${flash_attn_version} is already installed."
    else
        echo "flash-attn not found or outdated. Installing v${flash_attn_version}..."
        # Construct download URL
        WHEEL_FILENAME="flash_attn-${flash_attn_version}+cu${cu}torch${torch}cxx${cxx}abiFALSE-cp${cp}-cp${cp}-linux_x86_64.whl"
        DOWNLOAD_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${flash_attn_version}/${WHEEL_FILENAME}"

        if wget --continue --timeout=60 --no-check-certificate --tries=5 --waitretry=10 -O "${WHEEL_FILENAME}" "${DOWNLOAD_URL}"; then
            echo "Download successful: ${WHEEL_FILENAME}"
            # Install using uv (preferred), fallback to pip
            if command -v uv &> /dev/null; then
                uv pip install --break-system-packages --no-cache-dir "${WHEEL_FILENAME}"
            else
                pip install --break-system-packages --no-cache-dir "${WHEEL_FILENAME}"
            fi

            # Verify installation
            if pip show flash-attn > /dev/null 2>&1; then
                echo "Successfully installed flash-attn v$(get_installed_flash_attn_version)"

                rm -rf "${WHEEL_FILENAME}"
            else
                echo "Installation failed!"
                exit 1
            fi
        else
            echo "Failed to download: ${DOWNLOAD_URL}"
            exit 1
        fi
    fi
}

# Install grouped_gemm
# Install Megatron-LM CUDA dependencies
install_grouped_gemm() {
    # Get cuda|torch|python|cxx versions
    cu=$(nvcc --version | grep "Cuda compilation tools" | awk '{print $5}' | cut -d '.' -f 1)
    torch=$(pip show torch | grep Version | awk '{print $2}' | cut -d '+' -f 1 | cut -d '.' -f 1,2)
    cp=$(python3 --version | awk '{print $2}' | awk -F. '{print $1$2}')
    cxx=$(g++ --version | grep 'g++' | awk '{print $3}' | cut -d '.' -f 1)
    grouped_gemm_version="${GROUPED_GEMM_VERSION}"

    if [[ $(get_installed_grouped_gemm_version) == "${grouped_gemm_version}" ]]; then
        echo "grouped_gemm v${grouped_gemm_version} is already installed."
    else
        echo "grouped_gemm not found or outdated. Installing v${grouped_gemm_version}..."
        # Construct download URL
        WHEEL_FILENAME="nv_grouped_gemm-${grouped_gemm_version}+cu${cu}torch${torch}cxx${cxx}abiTRUE-cp${cp}-cp${cp}-linux_x86_64.whl"
        DOWNLOAD_URL="https://github.com/fanshiqing/grouped_gemm/releases/download/v${grouped_gemm_version}/${WHEEL_FILENAME}"

        if wget --continue --timeout=60 --no-check-certificate --tries=5 --waitretry=10 -O "${WHEEL_FILENAME}" "${DOWNLOAD_URL}"; then
            echo "Download successful: ${WHEEL_FILENAME}"
            # Install using uv (preferred), fallback to pip
            if command -v uv &> /dev/null; then
                uv pip install --break-system-packages --no-cache-dir "${WHEEL_FILENAME}"
            else
                pip install --break-system-packages --no-cache-dir "${WHEEL_FILENAME}"
            fi

            # Verify installation
            if pip show nv-grouped-gemm > /dev/null 2>&1; then
                echo "Successfully installed nv-grouped-gemm v$(get_installed_grouped_gemm_version)"

                rm -rf "${WHEEL_FILENAME}"
            else
                echo "Installation failed!"
                exit 1
            fi
        else
            echo "Failed to download: ${DOWNLOAD_URL}"
            exit 1
        fi
    fi

}

# apex train
# Install Megatron-LM CUDA dependencies
install_apex() {
    if pip show apex > /dev/null 2>&1; then
        apex_version=$(pip show apex | grep "^Version:" | awk '{print $2}')
        echo "apex v${apex_version} is already installed."
    else
        git clone https://github.com/NVIDIA/apex
        cd apex
        pip install -v \
            --disable-pip-version-check \
            --no-cache-dir \
            --no-build-isolation \
            --config-settings='--build-option=--cpp_ext' \
            --config-settings='--build-option=--cuda_ext' \
            ./
        cd ..
        rm -r ./apex
    fi
}

install_transformer_engine() {
    local installed_version
    installed_version=$(get_installed_transformer_engine)

    # Check if transformer_engine is already installed and matches the expected commit hash
    if [[ -n "$installed_version" ]]; then
        local current_commit_hash
        current_commit_hash="${installed_version#*+}"

        if [[ "$current_commit_hash" == "${TRANSFORMER_ENGINE_COMMIT}" ]]; then
            echo "transformer_engine ${installed_version} is already installed with the expected commit hash: ${TRANSFORMER_ENGINE_COMMIT}."
            return 0
        else
            echo "Installed transformer_engine version is ${installed_version}, but does not match the expected commit hash: ${TRANSFORMER_ENGINE_COMMIT}. Starting reinstallation..."
        fi
    else
        echo "transformer_engine is not installed. Starting installation..."
    fi

    git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
    cd TransformerEngine
    git checkout ${TRANSFORMER_ENGINE_COMMIT}
    pip install --no-build-isolation . -vvv
    cd ..
    rm -r ./TransformerEngine

}

install_or_verify_flag_gems() {
    if pip show flag_gems > /dev/null 2>&1; then
        local flag_gems_version
        flag_gems_version=$(pip show flag_gems | grep "^Version:" | awk '{print $2}')
        echo "grouped_gemm v${flag_gems_version} is already installed."
        echo "To install a different version, please manually execute:"
        echo "pip install git+https://github.com/FlagOpen/FlagGems.git@<desired_tag>"
    else
        echo "Installing grouped_gemm..."
        pip install git+https://github.com/FlagOpen/FlagGems.git@v3.0 || {
            echo "Installation failed. Please check your network connection and permissions."
            exit 1
        }
        echo "grouped_gemm has been successfully installed."
    fi
}

install_or_verify_vllm() {
    if pip show vllm > /dev/null 2>&1; then
        local flag_gems_version
        flag_gems_version=$(pip show vllm | grep "^Version:" | awk '{print $2}')
        echo "vllm ${flag_gems_version} is already installed."
        echo "To install a different version, please manually execute:"
        echo "python tools/patch/unpatch.py --backend vllm"
        echo "MAX_JOBS=$(nproc) pip install --no-build-isolation -v ./third_party/vllm/."
    else
        echo "Installing vllm..."
        MAX_JOBS=$(nproc) pip install --no-build-isolation -v ./third_party/vllm/. || {
            echo "Installation failed. Please check your network connection and permissions."
            exit 1
        }
        echo "vllm has been successfully installed."
    fi
}