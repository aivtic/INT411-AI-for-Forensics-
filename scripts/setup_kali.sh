#!/bin/bash

# INT411 AI for Forensics Lab - Kali Linux Setup Script
# This script sets up the lab environment specifically for Kali Linux

echo "=========================================="
echo "INT411 AI FOR FORENSICS - KALI LINUX SETUP"
echo "=========================================="
echo ""

# Check if running on Kali Linux
if ! grep -qi "kali" /etc/os-release; then
    echo "[!] Warning: This script is optimized for Kali Linux"
    echo "[!] Some features may not work on other distributions"
    echo ""
fi

# Check if running as root
if [ "$EUID" -eq 0 ]; then
   echo "[!] Please do not run this script as root"
   echo "[!] Run: bash scripts/setup_kali.sh"
   exit 1
fi

# Step 1: Update system
echo "[*] Step 1: Updating Kali Linux system packages..."
sudo apt update
sudo apt upgrade -y

# Step 2: Install Python and development tools
echo "[*] Step 2: Installing Python and development tools..."
sudo apt install -y python3 python3-pip python3-dev python3-venv
sudo apt install -y build-essential libssl-dev libffi-dev
sudo apt install -y git curl wget

# Step 3: Check Python version
echo "[*] Step 3: Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "[+] Python version: $python_version"

# Step 4: Create virtual environment
echo "[*] Step 4: Creating Python virtual environment..."
python3 -m venv int411_env
source int411_env/bin/activate

# Step 5: Upgrade pip
echo "[*] Step 5: Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Step 6: Install requirements
echo "[*] Step 6: Installing Python packages..."
echo "[*] This may take 10-15 minutes..."
pip install -r requirements.txt

# Step 7: Create necessary directories
echo "[*] Step 7: Creating directories..."
mkdir -p models
mkdir -p results
mkdir -p output

# Step 8: Verify installation
echo "[*] Step 8: Verifying installation..."
python3 -c "import numpy, pandas, sklearn, tensorflow; print('[+] All packages installed successfully')" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "[+] Installation verification passed"
else
    echo "[!] Installation verification failed"
    echo "[!] Try running: pip install -r requirements.txt --force-reinstall"
fi

# Step 9: Check for GPU support (optional)
echo "[*] Step 9: Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "[+] NVIDIA GPU detected"
    echo "[+] For GPU acceleration, run: pip install tensorflow[and-cuda]"
else
    echo "[-] No NVIDIA GPU detected"
    echo "[*] Lab will run on CPU (slower but still functional)"
fi

# Step 10: Display next steps
echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source int411_env/bin/activate"
echo ""
echo "To run the lab exercises:"
echo "  python code/01_exploratory_analysis.py"
echo "  python code/02_model_training.py"
echo "  python code/03_forensic_analysis.py"
echo ""
echo "For more information, see KALI_LINUX_SETUP.md"
echo ""
