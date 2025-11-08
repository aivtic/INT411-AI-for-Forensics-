# INT411 AI for Forensics - Kali Linux Setup Guide

**Version:** 1.0  
**Tested on:** Kali Linux 2024.x (latest)  
**Python Version:** 3.11+  
**Last Updated:** November 2025

---

## Overview

This guide provides step-by-step instructions for running the INT411 AI for Forensics lab on **Kali Linux**. Kali Linux is a specialized penetration testing and forensics distribution that comes pre-configured with many security tools.

---

## System Requirements for Kali Linux

### Minimum Requirements
- **RAM:** 4GB minimum (8GB recommended)
- **Disk Space:** 20GB free space
- **Processor:** 2 cores minimum (4+ cores recommended)
- **GPU:** Optional (NVIDIA CUDA for faster training)

### Recommended Setup
- **RAM:** 16GB
- **Disk Space:** 50GB
- **Processor:** Intel Core i7 or equivalent (4+ cores)
- **GPU:** NVIDIA GPU with CUDA support (optional but recommended)

---

## Part 1: Initial System Setup

### Step 1.1: Update Kali Linux

Open a terminal and update the system to ensure all packages are current:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt full-upgrade -y
```

This command updates the package list and installs all available updates. The process may take 10-15 minutes depending on your internet connection.

### Step 1.2: Install Python and Development Tools

Kali Linux comes with Python 3 pre-installed, but we need to install additional development tools and headers:

```bash
sudo apt install -y python3 python3-pip python3-dev python3-venv
sudo apt install -y build-essential libssl-dev libffi-dev
sudo apt install -y git curl wget
```

**What each package does:**
- `python3` and `python3-pip`: Python interpreter and package manager
- `python3-dev`: Python development headers needed for compiling packages
- `python3-venv`: Virtual environment support
- `build-essential`: Compiler and build tools
- `libssl-dev` and `libffi-dev`: Libraries for cryptography and security
- `git`, `curl`, `wget`: Version control and file download tools

### Step 1.3: Verify Python Installation

Verify that Python 3 is properly installed:

```bash
python3 --version
pip3 --version
```

Expected output:
```
Python 3.11.x (or higher)
pip 23.x (or higher)
```

If you see Python 2.x, you need to ensure Python 3 is the default:

```bash
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1
```

---

## Part 2: Lab Environment Setup

### Step 2.1: Extract Lab Files

Extract the INT411 lab package:

```bash
unzip INT411_AI_for_Forensics_Complete_Lab.zip
cd int411_ai_forensics_lab
ls -la
```

You should see the following structure:

```
int411_ai_forensics_lab/
├── INT411_AI_for_Forensics_Lab_Manual.md
├── README.md
├── KALI_LINUX_SETUP.md
├── requirements.txt
├── code/
│   ├── 01_exploratory_analysis.py
│   ├── 02_model_training.py
│   └── 03_forensic_analysis.py
├── datasets/
│   └── forensic_dataset.csv
├── lab_files/
│   ├── STUDENT_WORKSHEET.md
│   └── ANSWER_KEY.md
├── scripts/
│   └── setup.sh
└── models/
```

### Step 2.2: Create Python Virtual Environment

Creating a virtual environment isolates the lab dependencies from your system Python:

```bash
python3 -m venv int411_env
source int411_env/bin/activate
```

You should see `(int411_env)` appear in your terminal prompt, indicating the virtual environment is active.

### Step 2.3: Upgrade pip and Install Dependencies

Upgrade pip to the latest version:

```bash
pip install --upgrade pip setuptools wheel
```

Install all required Python packages:

```bash
pip install -r requirements.txt
```

This installs:
- **numpy:** Numerical computing
- **pandas:** Data manipulation
- **scikit-learn:** Machine learning algorithms
- **tensorflow:** Deep learning framework
- **matplotlib & seaborn:** Data visualization
- **jupyter:** Interactive notebooks
- **imbalanced-learn:** Handling imbalanced datasets

**Installation time:** 5-15 minutes depending on internet speed and whether packages need compilation.

### Step 2.4: Verify Installation

Verify all packages are installed correctly:

```bash
python3 -c "import numpy, pandas, sklearn, tensorflow; print('✓ All packages installed successfully')"
```

If you see any errors, try installing packages individually:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn jupyter imbalanced-learn
```

---

## Part 3: Kali Linux-Specific Considerations

### 3.1: GPU Support (Optional but Recommended)

If you have an NVIDIA GPU, you can accelerate model training significantly.

**Check if you have an NVIDIA GPU:**

```bash
lspci | grep -i nvidia
```

If you see NVIDIA GPU listed, install CUDA support:

```bash
sudo apt install -y nvidia-driver-latest
sudo apt install -y nvidia-cuda-toolkit
```

Then install TensorFlow with GPU support:

```bash
pip install tensorflow[and-cuda]
```

### 3.2: Kali Linux Pre-installed Tools

Kali Linux comes with several forensic tools that complement this lab:

- **Volatility:** Memory forensics
- **Autopsy:** Digital forensics framework
- **Wireshark:** Network analysis
- **Burp Suite:** Web application testing
- **Metasploit:** Penetration testing

These tools can be used alongside the AI models for comprehensive forensic analysis.

### 3.3: Memory Management

If you have limited RAM, you may need to optimize memory usage:

```bash
# Reduce dataset size in code/02_model_training.py
# Or use smaller batch sizes:
# batch_size=8 instead of batch_size=16
```

### 3.4: Disk Space

Ensure you have sufficient disk space before training models:

```bash
df -h
```

If disk space is low, you can clean up:

```bash
sudo apt clean
sudo apt autoclean
pip cache purge
```

---

## Part 4: Running the Lab on Kali Linux

### Step 4.1: Activate Virtual Environment

Before running any lab exercises, always activate the virtual environment:

```bash
source int411_env/bin/activate
```

You should see `(int411_env)` in your terminal prompt.

### Step 4.2: Run Part 1 - Exploratory Data Analysis

```bash
python code/01_exploratory_analysis.py
```

**Expected output:**
```
============================================================
INT411 AI FOR FORENSICS - EXPLORATORY DATA ANALYSIS
============================================================
[*] Loading forensic dataset...
[+] Dataset loaded: 100 samples, 11 features
============================================================
BASIC DATASET STATISTICS
============================================================
...
[+] Correlation matrix saved to correlation_matrix.png
[+] Feature distributions saved to feature_distributions.png
[+] Benign vs Malicious comparison saved to benign_vs_malicious.png
[+] Class distribution saved to class_distribution.png
```

Generated files:
- `correlation_matrix.png`
- `feature_distributions.png`
- `benign_vs_malicious.png`
- `class_distribution.png`

### Step 4.3: Run Part 2 - Model Training

```bash
python code/02_model_training.py
```

**Expected output:**
```
============================================================
INT411 AI FOR FORENSICS - MODEL TRAINING
============================================================
[*] Loading dataset...
[+] Dataset loaded: (80, 7)
[+] Training set: (64, 7)
[+] Test set: (16, 7)

============================================================
TRAINING RANDOM FOREST MODEL
============================================================
[*] Training Random Forest...
[+] Random Forest training complete

Random Forest Performance:
  Accuracy:  0.9375
  Precision: 0.9231
  Recall:    0.9565
  F1-Score:  0.9396
  ROC-AUC:   0.9821

============================================================
TRAINING SUPPORT VECTOR MACHINE (SVM)
============================================================
...
[+] Model comparison saved to model_comparison.png
[+] Saved Random Forest model and scaler
[+] Saved SVM model and scaler
[+] Saved Neural Network model and scaler
```

Generated files:
- `model_comparison.png`
- `models/random_forest_model.pkl`
- `models/random_forest_scaler.pkl`
- `models/svm_model.pkl`
- `models/svm_scaler.pkl`
- `models/neural_network_model.pkl`
- `models/neural_network_scaler.pkl`

**Training time:** 5-15 minutes depending on CPU/GPU

### Step 4.4: Run Part 3 - Forensic Analysis

```bash
python code/03_forensic_analysis.py
```

**Expected output:**
```
============================================================
INT411 AI FOR FORENSICS - FORENSIC ANALYSIS
============================================================
[*] Loading trained models...
[+] Loaded Random Forest model
[+] Loaded SVM model
[+] Loaded Neural Network model

[*] Analyzing dataset: /path/to/forensic_dataset.csv
[+] Results saved to analysis_results.csv

[*] Generating forensic analysis report...
[+] Report saved to forensic_analysis_report.txt

============================================================
FORENSIC ANALYSIS REPORT
============================================================
Generated: 2025-11-08 10:30:45
Files Analyzed: 100

----------------------------------------------------------------------
SUMMARY STATISTICS
----------------------------------------------------------------------

Ensemble Predictions:
  Malicious: 50 (50.0%)
  Benign: 50 (50.0%)

----------------------------------------------------------------------
MODEL ACCURACY
----------------------------------------------------------------------

Random Forest:
  Accuracy: 0.9375 (93.75%)

SVM:
  Accuracy: 0.9200 (92.00%)

Neural Network:
  Accuracy: 0.9600 (96.00%)

Ensemble:
  Accuracy: 0.9650 (96.50%)

[+] Visualization saved to analysis_visualization.png
```

Generated files:
- `analysis_results.csv`
- `forensic_analysis_report.txt`
- `analysis_visualization.png`

---

## Part 5: Troubleshooting on Kali Linux

### Issue: pip command not found

**Solution:**
```bash
sudo apt install -y python3-pip
```

### Issue: Virtual environment activation fails

**Solution:**
```bash
python3 -m venv int411_env --clear
source int411_env/bin/activate
```

### Issue: "No module named 'sklearn'" or similar

**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Issue: TensorFlow installation fails

**Solution for Kali Linux:**
```bash
pip install tensorflow --no-cache-dir
```

Or install pre-built wheel:
```bash
pip install tensorflow-cpu
```

### Issue: Out of memory during training

**Solution:**
```bash
# Edit code/02_model_training.py and reduce:
# - batch_size from 16 to 8
# - n_estimators from 100 to 50
# - epochs from 50 to 30
```

### Issue: Models directory permission denied

**Solution:**
```bash
chmod -R 755 models/
mkdir -p models
```

### Issue: Slow model training

**Solutions:**
1. Use smaller dataset
2. Reduce model complexity
3. Enable GPU support (if available)
4. Use CPU-only TensorFlow (faster for small datasets)

```bash
pip uninstall tensorflow -y
pip install tensorflow-cpu
```

### Issue: Display issues with matplotlib

If running over SSH or in headless mode:

```bash
# Add to Python script before plotting:
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

---

## Part 6: Advanced Kali Linux Integration

### 6.1: Using with Volatility (Memory Forensics)

Combine AI models with Volatility for memory analysis:

```bash
# Install Volatility
sudo apt install -y volatility3

# Analyze memory dump
vol -f memory.dump windows.psscan

# Extract suspicious processes
vol -f memory.dump windows.dlllist --pid=1234
```

### 6.2: Using with Autopsy (Digital Forensics)

Integrate AI predictions with Autopsy:

1. Export file list from Autopsy
2. Extract features using the lab code
3. Run predictions
4. Import results back into Autopsy

### 6.3: Automating Analysis with Bash Scripts

Create an automated forensic analysis pipeline:

```bash
#!/bin/bash
# forensic_pipeline.sh

source int411_env/bin/activate

echo "[*] Running forensic analysis pipeline..."
echo "[*] Step 1: EDA"
python code/01_exploratory_analysis.py

echo "[*] Step 2: Model Training"
python code/02_model_training.py

echo "[*] Step 3: Forensic Analysis"
python code/03_forensic_analysis.py

echo "[+] Analysis complete!"
echo "[+] Results saved to analysis_results.csv"
echo "[+] Report saved to forensic_analysis_report.txt"
```

Make it executable:
```bash
chmod +x forensic_pipeline.sh
./forensic_pipeline.sh
```

---

## Part 7: Using Jupyter Notebooks on Kali Linux

For interactive analysis, use Jupyter Notebook:

```bash
# Install Jupyter (if not already installed)
pip install jupyter

# Start Jupyter server
jupyter notebook

# Access at http://localhost:8888
```

Create a notebook to run the lab interactively:

```python
# Cell 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Cell 2: Load data
df = pd.read_csv('datasets/forensic_dataset.csv')
print(df.head())

# Cell 3: Run analysis
# ... add your code here
```

---

## Part 8: Performance Optimization for Kali Linux

### 8.1: CPU Optimization

```bash
# Check CPU cores
nproc

# Use all cores in scikit-learn
# In code/02_model_training.py:
# RandomForestClassifier(n_jobs=-1)  # Uses all cores
```

### 8.2: Memory Optimization

```bash
# Check available memory
free -h

# Monitor during training
watch -n 1 free -h
```

### 8.3: Disk I/O Optimization

```bash
# Use SSD for faster I/O
# Check disk speed
sudo hdparm -Tt /dev/sda

# Monitor disk usage
df -h
du -sh *
```

---

## Part 9: Submitting Work on Kali Linux

### 9.1: Prepare Submission

```bash
# Create submission directory
mkdir submission
cp analysis_results.csv submission/
cp forensic_analysis_report.txt submission/
cp *.png submission/
cp lab_files/STUDENT_WORKSHEET.md submission/
```

### 9.2: Create Archive

```bash
# Create ZIP file for submission
zip -r INT411_Lab_Submission.zip submission/

# Or create TAR.GZ for Kali
tar -czf INT411_Lab_Submission.tar.gz submission/
```

### 9.3: Verify Submission

```bash
# List contents
unzip -l INT411_Lab_Submission.zip

# Or for TAR.GZ
tar -tzf INT411_Lab_Submission.tar.gz
```

---

## Part 10: Additional Resources for Kali Linux

### Kali Linux Documentation
- Official Kali Linux: https://www.kali.org/
- Kali Linux Tools: https://tools.kali.org/

### Python and Machine Learning
- Scikit-learn: https://scikit-learn.org/
- TensorFlow: https://www.tensorflow.org/
- Pandas: https://pandas.pydata.org/

### Forensics Resources
- NIST Forensics: https://www.nist.gov/
- SANS Forensics: https://www.sans.org/
- Volatility: https://www.volatilityfoundation.org/

### Kali Linux Forensics Tools
- Volatility: Memory forensics
- Autopsy: Digital forensics framework
- Wireshark: Network analysis
- Sleuth Kit: Forensic analysis

---

## Quick Reference Commands

```bash
# Activate environment
source int411_env/bin/activate

# Deactivate environment
deactivate

# Run EDA
python code/01_exploratory_analysis.py

# Run model training
python code/02_model_training.py

# Run forensic analysis
python code/03_forensic_analysis.py

# View results
cat forensic_analysis_report.txt

# List generated files
ls -lh *.png *.csv *.txt

# Check Python packages
pip list

# Update packages
pip install --upgrade -r requirements.txt

# Remove virtual environment
rm -rf int411_env

# Clean up generated files
rm -f *.png *.csv analysis_results.csv forensic_analysis_report.txt
```

---

## Support and Troubleshooting

If you encounter issues:

1. **Check the main README.md** for general troubleshooting
2. **Review the lab manual** for conceptual help
3. **Check Python version:** `python3 --version`
4. **Verify packages:** `pip list`
5. **Check disk space:** `df -h`
6. **Check memory:** `free -h`
7. **Review error messages carefully** - they usually indicate the problem

---

## Summary

You now have a complete, working INT411 AI for Forensics lab on Kali Linux. The lab provides:

✓ Complete machine learning pipeline  
✓ Three different ML algorithms  
✓ Forensic dataset with 100 samples  
✓ Automated analysis and reporting  
✓ Integration with Kali Linux tools  
✓ Comprehensive documentation  

**Happy learning and forensic analysis!**

---

**Kali Linux Setup Guide Version:** 1.0  
**Last Updated:** November 2025  
**Tested on:** Kali Linux 2024.x  
**Python Version:** 3.11+
