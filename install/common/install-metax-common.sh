#!/bin/bash

python -m pip install --upgrade pip

print_help() {
cat << EOF
Usage: $0 [OPTIONS]

Options:
  --torch-ver <version>          Specify the PyTorch version (e.g., "2.6.0+metax3.0.0.3")
  --torchaudio-ver <version>     Specify the TorchAudio version (e.g., "2.4.1+metax3.0.0.3")
  --torchvision-ver <version>    Specify the TorchVision version (e.g., "0.15.1+metax3.0.0.3")
  -h|--help                      Show this help message and exit

Example:
  $0 --torch-ver "2.6.0+metax3.0.0.3" \
    --torchaudio-ver "2.4.1+metax3.0.0.3" 
    --torchvision-ver "0.15.1+metax3.0.0.3" 
EOF
}

# Define variables for clarity and maintainability
# Initialize the variable
PYTORCH_VER="2.6.0+metax3.0.0.3"
TORCHAUDIO_VER="2.4.1+metax3.0.0.3"
TORCHVISION_VER="0.15.1+metax3.0.0.3"

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --torch-ver) PYTORCH_VER="$2"; shift ;;
        --torchaudio-ver) TORCHAUDIO_VER="$2"; shift ;;
        --torchvision-ver) TORCHVISION_VER="$2"; shift ;;
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
            -i https://repos.metax-tech.com/r/maca-pypi/simple \
            --trusted-host repos.metax-tech.com || { echo "Installation failed. Please check network connection and permission settings."; exit 1; }
    else
        echo "Using 'pip' for installation..."
        pip install \
            torch=="$PYTORCH_VER" \
            torchaudio=="$TORCHAUDIO_VER" \
            torchvision=="$TORCHVISION_VER" \
            -i https://repos.metax-tech.com/r/maca-pypi/simple \
            --trusted-host repos.metax-tech.com || { echo "Installation failed. Please check network connection and permission settings."; exit 1; }
    fi
}
