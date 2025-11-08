#!/bin/bash

# INT411 AI for Forensics Lab - Setup Script
# This script sets up the lab environment and installs all dependencies

echo "=========================================="
echo "INT411 AI FOR FORENSICS - LAB SETUP"
echo "=========================================="

# Check Python version
echo "[*] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "[+] Python version: $python_version"

# Create virtual environment
echo "[*] Creating virtual environment..."
python3 -m venv int411_env
source int411_env/bin/activate

# Upgrade pip
echo "[*] Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "[*] Installing required packages..."
pip install -r requirements.txt

# Create necessary directories
echo "[*] Creating directories..."
mkdir -p models
mkdir -p results

# Verify installation
echo "[*] Verifying installation..."
python3 -c "import numpy, pandas, sklearn, tensorflow; print('[+] All packages installed successfully')"

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
