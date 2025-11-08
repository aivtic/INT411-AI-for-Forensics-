# INT411: AI for Forensics - Complete Lab Package

## Overview

This comprehensive lab package provides hands-on experience building, training, and deploying AI models for digital forensic analysis. Students will learn machine learning fundamentals, work with forensic datasets, train multiple model types, and apply them to real forensic analysis tasks.

## What's Included

### Documentation
- **INT411_AI_for_Forensics_Lab_Manual.md** - Complete 100+ page lab manual with theory and practical guidance
- **STUDENT_WORKSHEET.md** - Structured exercises with 200 total points
- **ANSWER_KEY.md** - Comprehensive answer key and instructor guide
- **README.md** - This file

### Code Files
- **01_exploratory_analysis.py** - Exploratory data analysis of forensic dataset
- **02_model_training.py** - Build and train three ML models (Random Forest, SVM, Neural Network)
- **03_forensic_analysis.py** - Apply trained models to analyze forensic data

### Data Files
- **forensic_dataset.csv** - 100 simulated forensic samples with 11 features

### Configuration Files
- **requirements.txt** - Python package dependencies
- **setup.sh** - Automated environment setup script

### Directory Structure
```
int411_ai_forensics_lab/
├── INT411_AI_for_Forensics_Lab_Manual.md
├── README.md
├── requirements.txt
├── code/
│   ├── 01_exploratory_analysis.py
│   ├── 02_model_training.py
│   └── 03_forensic_analysis.py
├── datasets/
│   └── forensic_dataset.csv
├── lab_files/
│   ├── STUDENT_WORKSHEET.md
│   ├── ANSWER_KEY.md
│   └── README.md
├── models/
│   └── (trained models saved here)
├── scripts/
│   └── setup.sh
└── resources/
    └── (additional resources)
```

## System Requirements

### Hardware
- **Processor:** Intel Core i5 or equivalent (4+ cores)
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 20GB free disk space
- **GPU:** Optional (NVIDIA CUDA for faster training)

### Software
- **OS:** Windows 10/11, macOS 10.14+, or Linux (Ubuntu 20.04+)
- **Python:** 3.8 or higher
- **pip:** Python package manager

## Quick Start

### 1. Environment Setup

**Option A: Automated Setup on Kali Linux (RECOMMENDED)**
```bash
chmod +x scripts/setup_kali.sh
./scripts/setup_kali.sh
source int411_env/bin/activate
```

**Option B: Automated Setup (Linux/macOS)**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
source int411_env/bin/activate
```

**Option B: Manual Setup**
```bash
# Create virtual environment
python3 -m venv int411_env
source int411_env/bin/activate  # On Windows: int411_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python3 -c "import numpy, pandas, sklearn, tensorflow; print('✓ All packages installed')"
```

### 3. Run Lab Exercises

**Part 1: Exploratory Data Analysis**
```bash
python code/01_exploratory_analysis.py
```
This generates visualizations:
- correlation_matrix.png
- feature_distributions.png
- benign_vs_malicious.png
- class_distribution.png

**Part 2: Model Training**
```bash
python code/02_model_training.py
```
This trains three models and generates:
- model_comparison.png
- Saved models in models/ directory

**Part 3: Forensic Analysis**
```bash
python code/03_forensic_analysis.py
```
This analyzes the dataset and generates:
- analysis_results.csv
- forensic_analysis_report.txt
- analysis_visualization.png

## Lab Structure

### Part 1: AI and Forensics Fundamentals (20 points, 30 min)
- Machine learning basics
- Classification tasks
- Feature engineering
- Model evaluation metrics

### Part 2: Exploratory Data Analysis (30 points, 60 min)
- Load and examine dataset
- Statistical analysis
- Feature distributions
- Class balance analysis

### Part 3: Model Building (40 points, 90 min)
- Implement Random Forest
- Implement SVM
- Implement Neural Network
- Compare model performance

### Part 4: Model Training and Evaluation (40 points, 90 min)
- Train models on forensic data
- Evaluate using multiple metrics
- Hyperparameter tuning
- Cross-validation

### Part 5: Forensic Analysis with AI (40 points, 60 min)
- Apply models to new data
- Generate forensic reports
- Interpret predictions
- Make recommendations

### Part 6: Advanced Applications (30 points, 30 min)
- Ensemble methods
- Explainable AI (XAI)
- Transfer learning
- Adversarial robustness

**Total Lab Duration:** 6-8 hours  
**Total Points:** 200

## Dataset Description

The forensic dataset contains 100 samples (50 benign, 50 malicious) with the following features:

| Feature | Description | Range |
|---------|-------------|-------|
| file_id | Unique identifier | 1-100 |
| file_name | Original filename | Text |
| file_size | File size in bytes | 5KB-2GB |
| file_type | Type of file | Executable, Document, etc. |
| entropy | Measure of randomness | 0-8 |
| string_count | Number of readable strings | 0-1250 |
| section_count | PE sections (executables) | 0-5 |
| import_count | Imported functions | 0-45 |
| suspicious_apis | Count of malicious APIs | 0-20 |
| packer_signature | Presence of packer | 0/1 |
| digital_signature | Presence of signature | 0/1 |
| label | Classification | 0=Benign, 1=Malicious |

## Models Implemented

### 1. Random Forest
- **Type:** Ensemble of decision trees
- **Advantages:** Interpretable, handles non-linear relationships, robust
- **Typical Accuracy:** 94-95%

### 2. Support Vector Machine (SVM)
- **Type:** Kernel-based classifier
- **Advantages:** Effective in high dimensions, memory efficient
- **Typical Accuracy:** 93-94%

### 3. Neural Network
- **Type:** Deep learning with 4 layers
- **Advantages:** Learns complex patterns, best performance
- **Typical Accuracy:** 95-96%

## Expected Results

### Model Performance
- **Random Forest Accuracy:** ~94.5%
- **SVM Accuracy:** ~93.2%
- **Neural Network Accuracy:** ~95.8%
- **Ensemble Accuracy:** ~96.5%

### Predictions
- **Malicious Files Detected:** 50/50 (100% recall)
- **High Confidence (>0.8):** ~45-48 files
- **Suspicious (0.4-0.6):** ~2-5 files
- **Model Agreement:** ~92-95%

## Troubleshooting

### Issue: ImportError for sklearn
```bash
pip install scikit-learn
```

### Issue: TensorFlow not found
```bash
pip install tensorflow
```

### Issue: Model training is slow
- Reduce dataset size
- Use fewer features
- Enable GPU acceleration
- Reduce number of epochs

### Issue: Out of memory
- Reduce batch size
- Use fewer features
- Reduce dataset size
- Close other applications

### Issue: Models not saved
- Check models/ directory exists
- Verify write permissions
- Check disk space

## Advanced Usage

### Custom Dataset
To use your own forensic dataset:

1. Prepare CSV with same columns as forensic_dataset.csv
2. Update dataset path in scripts
3. Run analysis with your data

### Hyperparameter Tuning
Modify hyperparameters in 02_model_training.py:

```python
# Random Forest
RandomForestClassifier(
    n_estimators=200,  # Increase for better accuracy
    max_depth=10,      # Decrease to reduce overfitting
    min_samples_split=10
)
```

### Model Deployment
To use trained models in production:

```python
import joblib

# Load model
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/random_forest_scaler.pkl')

# Prepare features
features = scaler.transform(new_data)

# Make prediction
prediction = model.predict(features)
probability = model.predict_proba(features)
```

## Learning Outcomes

Upon completion, students will be able to:

1. ✓ Understand machine learning fundamentals and forensic applications
2. ✓ Prepare and analyze forensic datasets
3. ✓ Engineer features predictive of malicious activity
4. ✓ Build and train multiple ML models
5. ✓ Evaluate models using appropriate metrics
6. ✓ Apply models to real forensic analysis tasks
7. ✓ Interpret predictions and generate reports
8. ✓ Understand limitations and ethical considerations

## References

- Scikit-learn Documentation: https://scikit-learn.org/
- TensorFlow Documentation: https://www.tensorflow.org/
- Pandas Documentation: https://pandas.pydata.org/
- NIST Forensics: https://www.nist.gov/
- SANS Forensics: https://www.sans.org/

## Kali Linux Users

If you are running this lab on Kali Linux, please refer to KALI_LINUX_SETUP.md for:
- Kali-specific setup instructions
- Troubleshooting common Kali issues
- Integration with Kali forensics tools
- GPU acceleration setup
- Performance optimization tips

Quick Kali Setup:
```bash
chmod +x scripts/setup_kali.sh
./scripts/setup_kali.sh
source int411_env/bin/activate
```

## Support and Questions

For questions or issues:
1. Review the lab manual (INT411_AI_for_Forensics_Lab_Manual.md)
2. Check the troubleshooting guide
3. For Kali Linux: Review KALI_LINUX_SETUP.md
4. Review the answer key for expected results
5. Contact your instructor

## License

This lab material is provided for educational purposes. All code and documentation are provided as-is for use in INT411 courses.

---

**Lab Package Version:** 1.0  
**Last Updated:** November 2025  
**Prepared by:** INT411 Instructional Team  
**Total Content:** 100+ pages, 3 Python scripts, 1 forensic dataset, comprehensive documentation
