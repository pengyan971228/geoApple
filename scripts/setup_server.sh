#!/usr/bin/env bash
# =============================================================================
# GeoApple-Seg Server Setup Script
# Target: Fresh Linux server with NVIDIA RTX 5090
# Usage:  chmod +x setup_server.sh && sudo ./setup_server.sh
# =============================================================================
set -uo pipefail

# ---- Configuration ----------------------------------------------------------
PYTHON_VERSION="3.12"
CUDA_VERSION="12.8"                # RTX 5090 (Blackwell) requires CUDA 12.8+
CONDA_ENV_NAME="geo_apple"
PROJECT_DIR="$HOME/ml/geo_apple_detection"
LOG_FILE="/tmp/setup_server_$(date +%Y%m%d_%H%M%S).log"

# ---- Colors -----------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $*" | tee -a "$LOG_FILE"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*" | tee -a "$LOG_FILE"; exit 1; }

# ---- Pre-flight checks ------------------------------------------------------
check_root() {
    if [[ $EUID -ne 0 ]]; then
        err "This script must be run as root (use sudo)"
    fi
    REAL_USER="${SUDO_USER:-$USER}"
    REAL_HOME=$(eval echo "~$REAL_USER")
    log "Running as root, real user: $REAL_USER, home: $REAL_HOME"
}

check_os() {
    if [[ ! -f /etc/os-release ]]; then
        err "Cannot detect OS. Only Ubuntu 22.04/24.04 is supported."
    fi
    source /etc/os-release
    log "Detected OS: $PRETTY_NAME"
    if [[ "$ID" != "ubuntu" ]]; then
        warn "This script is optimized for Ubuntu. Proceed with caution on $ID."
    fi
}

check_gpu() {
    if lspci | grep -qi nvidia; then
        log "NVIDIA GPU detected: $(lspci | grep -i nvidia | head -1)"
    else
        warn "No NVIDIA GPU detected via lspci. Driver installation may fail."
    fi
}

# =============================================================================
# Step 1: System packages
# =============================================================================
install_system_packages() {
    log "Step 1/7: Installing system packages..."
    apt-get update -qq
    apt-get install -y -qq \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        unzip \
        htop \
        tmux \
        vim \
        tree \
        software-properties-common \
        ca-certificates \
        gnupg \
        lsb-release \
        pkg-config \
        libssl-dev \
        libffi-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
        libglib2.0-0
    log "System packages installed."
}

# =============================================================================
# Step 2: NVIDIA Driver
# =============================================================================
install_nvidia_driver() {
    log "Step 2/7: Installing NVIDIA driver..."

    # Always add NVIDIA package repository (needed for CUDA toolkit in Step 3)
    local distro
    distro="ubuntu$(lsb_release -rs | tr -d '.')"
    local arch
    arch="$(uname -m)"

    wget -qO /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${arch}/cuda-keyring_1.1-1_all.deb"
    dpkg -i /tmp/cuda-keyring.deb 2>&1
    apt-get update -qq
    log "NVIDIA package repository added."

    if command -v nvidia-smi &> /dev/null; then
        local driver_ver
        driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        log "NVIDIA driver already installed: v${driver_ver}"
        local major_ver
        major_ver=$(echo "$driver_ver" | cut -d. -f1)
        if [[ "$major_ver" -ge 570 ]]; then
            log "Driver version sufficient for RTX 5090. Skipping driver install."
            return 0
        else
            warn "Driver v${driver_ver} may be too old for RTX 5090. Upgrading..."
        fi
    fi

    # Install latest driver (570+ for Blackwell)
    apt-get install -y -qq nvidia-driver-570 2>&1 || \
        apt-get install -y -qq nvidia-driver-565 2>&1 || \
        err "Failed to install NVIDIA driver. Install manually: sudo ubuntu-drivers install"

    log "NVIDIA driver installed. Reboot required after setup completes."
}

# =============================================================================
# Step 3: CUDA Toolkit
# =============================================================================
install_cuda() {
    log "Step 3/7: Installing CUDA ${CUDA_VERSION} toolkit..."

    if command -v nvcc &> /dev/null; then
        local cuda_ver
        cuda_ver=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        log "CUDA already installed: v${cuda_ver}"
        if [[ "$cuda_ver" == "${CUDA_VERSION}"* ]]; then
            log "CUDA version matches. Skipping."
            return 0
        else
            warn "CUDA v${cuda_ver} found, installing v${CUDA_VERSION}..."
        fi
    fi

    apt-get install -y -qq \
        "cuda-toolkit-${CUDA_VERSION/./-}" \
        2>&1 || err "Failed to install CUDA toolkit ${CUDA_VERSION}"

    # cuDNN (required for deep learning)
    apt-get install -y -qq \
        "libcudnn9-cuda-${CUDA_VERSION%%.*}" \
        "libcudnn9-dev-cuda-${CUDA_VERSION%%.*}" \
        2>&1 || warn "cuDNN installation failed. Install manually if needed."

    # Set up environment variables
    local cuda_env_file="/etc/profile.d/cuda.sh"
    cat > "$cuda_env_file" << 'CUDA_EOF'
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
CUDA_EOF
    chmod +x "$cuda_env_file"

    log "CUDA ${CUDA_VERSION} installed. Environment set in ${cuda_env_file}"
}

# =============================================================================
# Step 4: Miniconda
# =============================================================================
install_miniconda() {
    log "Step 4/7: Installing Miniconda..."

    local conda_dir="${REAL_HOME}/miniconda3"

    if [[ -d "$conda_dir" ]]; then
        log "Miniconda already installed at ${conda_dir}. Skipping."
        return 0
    fi

    local installer="/tmp/Miniconda3-latest-Linux-x86_64.sh"
    wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O "$installer"
    chmod +x "$installer"

    # Install as real user
    sudo -u "$REAL_USER" bash "$installer" -b -p "$conda_dir"
    rm -f "$installer"

    # Initialize conda for the real user's shell
    sudo -u "$REAL_USER" "$conda_dir/bin/conda" init bash
    if [[ -f "${REAL_HOME}/.zshrc" ]]; then
        sudo -u "$REAL_USER" "$conda_dir/bin/conda" init zsh
    fi

    # Disable auto-activate base
    sudo -u "$REAL_USER" "$conda_dir/bin/conda" config --set auto_activate_base false

    log "Miniconda installed at ${conda_dir}"
}

# =============================================================================
# Step 5: uv (fast Python package manager)
# =============================================================================
install_uv() {
    log "Step 5/7: Installing uv package manager..."

    if sudo -u "$REAL_USER" bash -c 'command -v uv' &> /dev/null; then
        local uv_ver
        uv_ver=$(sudo -u "$REAL_USER" uv --version 2>/dev/null)
        log "uv already installed: ${uv_ver}. Skipping."
        return 0
    fi

    sudo -u "$REAL_USER" bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' \
        2>&1

    log "uv installed."
}

# =============================================================================
# Step 6: Create conda environment with PyTorch
# =============================================================================
setup_conda_env() {
    log "Step 6/7: Creating conda environment '${CONDA_ENV_NAME}'..."

    local conda_bin="${REAL_HOME}/miniconda3/bin/conda"

    # Check if env exists
    if sudo -u "$REAL_USER" "$conda_bin" env list | grep -q "$CONDA_ENV_NAME"; then
        log "Conda env '${CONDA_ENV_NAME}' already exists. Skipping creation."
    else
        sudo -u "$REAL_USER" "$conda_bin" create -n "$CONDA_ENV_NAME" \
            "python=${PYTHON_VERSION}" -y -q
        log "Conda env '${CONDA_ENV_NAME}' created with Python ${PYTHON_VERSION}"
    fi

    # Install PyTorch + core packages via pip inside conda env
    local pip_bin="${REAL_HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/pip"

    log "Installing PyTorch (CUDA ${CUDA_VERSION})..."
    sudo -u "$REAL_USER" "$pip_bin" install -q \
        torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION/.}"

    log "Installing ML packages..."
    sudo -u "$REAL_USER" "$pip_bin" install -q \
        ultralytics \
        opencv-python-headless \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
        Pillow \
        tqdm \
        wandb \
        tensorboard \
        albumentations \
        pycocotools \
        hydra-core \
        omegaconf \
        rich

    log "Installing dev tools..."
    sudo -u "$REAL_USER" "$pip_bin" install -q \
        ruff \
        mypy \
        pytest \
        ipython \
        jupyter \
        notebook

    log "Conda environment '${CONDA_ENV_NAME}' ready."
}

# =============================================================================
# Step 7: Verify installation
# =============================================================================
verify_installation() {
    log "Step 7/7: Verifying installation..."

    local python_bin="${REAL_HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python"
    local errors=0

    # Python
    if sudo -u "$REAL_USER" "$python_bin" --version &> /dev/null; then
        log "  Python: $(sudo -u "$REAL_USER" "$python_bin" --version)"
    else
        warn "  Python: FAILED"; ((errors++))
    fi

    # PyTorch + CUDA
    local torch_check
    torch_check=$(sudo -u "$REAL_USER" "$python_bin" -c "
import torch
ver = torch.__version__
cuda = torch.cuda.is_available()
gpu = torch.cuda.get_device_name(0) if cuda else 'N/A'
print(f'PyTorch {ver} | CUDA available: {cuda} | GPU: {gpu}')
" 2>/dev/null) || torch_check="FAILED"
    log "  ${torch_check}"
    if [[ "$torch_check" == *"FAILED"* ]]; then ((errors++)); fi

    # ultralytics
    local yolo_check
    yolo_check=$(sudo -u "$REAL_USER" "$python_bin" -c "
import ultralytics; print(f'Ultralytics {ultralytics.__version__}')
" 2>/dev/null) || yolo_check="FAILED"
    log "  ${yolo_check}"

    # uv
    if sudo -u "$REAL_USER" bash -c 'command -v uv' &> /dev/null; then
        log "  uv: $(sudo -u "$REAL_USER" uv --version 2>/dev/null)"
    else
        warn "  uv: not found in PATH"
    fi

    # NVIDIA
    if command -v nvidia-smi &> /dev/null; then
        log "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
    else
        warn "  nvidia-smi: not available (reboot may be needed)"
    fi

    echo ""
    if [[ $errors -eq 0 ]]; then
        log "=========================================="
        log "  Setup complete! All checks passed."
        log "=========================================="
    else
        warn "Setup completed with ${errors} warning(s). Check log: ${LOG_FILE}"
    fi

    echo ""
    log "Quick start:"
    log "  conda activate ${CONDA_ENV_NAME}"
    log "  python -c 'import torch; print(torch.cuda.get_device_name(0))'"
    echo ""
    log "If NVIDIA driver was freshly installed, reboot first:"
    log "  sudo reboot"
    echo ""
    log "Full log: ${LOG_FILE}"
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo "============================================"
    echo " GeoApple-Seg Server Environment Setup"
    echo " Target GPU: NVIDIA RTX 5090 (Blackwell)"
    echo " CUDA: ${CUDA_VERSION} | Python: ${PYTHON_VERSION}"
    echo "============================================"
    echo ""

    check_root
    check_os
    check_gpu

    install_system_packages
    install_nvidia_driver
    install_cuda
    install_miniconda
    install_uv
    setup_conda_env
    verify_installation
}

main "$@"
