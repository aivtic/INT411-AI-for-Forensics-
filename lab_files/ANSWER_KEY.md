# INT411 AI for Forensics - Answer Key and Instructor Guide

**Version:** 1.0  
**Last Updated:** November 2025

---

## Part 1: AI and Forensics Fundamentals - Answer Key (20 points)

### Question 1.1: Machine Learning Basics (5 points)

**Expected Answer:**

**Supervised Learning:** Learning from labeled data where the correct answer is known.
- Example 1: Classifying files as benign or malicious (label = file classification)
- Example 2: Predicting malware family from file features (label = malware family)

**Unsupervised Learning:** Learning patterns from unlabeled data without knowing correct answers.
- Example 1: Clustering similar malware samples to identify variants
- Example 2: Anomaly detection in system logs to find suspicious activity

**Grading Rubric:**
- 5 points: Clear definitions with appropriate forensic examples
- 4 points: Good definitions with minor gaps
- 3 points: Adequate definitions with some confusion
- 2 points: Basic understanding with significant gaps
- 1 point: Incomplete or incorrect understanding

### Question 1.2: Classification Task (5 points)

**Expected Answer:**

**False Positives Consequences:**
- Legitimate files quarantined unnecessarily
- System performance degradation
- User frustration and loss of productivity
- Potential data loss if files are deleted

**False Negatives Consequences:**
- Malware remains undetected on the system
- Potential data breach or system compromise
- Attacker maintains persistence
- Investigation may be incomplete

**More Problematic:** False negatives are generally more problematic in forensics because undetected malware poses active security risks, whereas false positives are manageable through manual review.

**Grading Rubric:**
- 5 points: Comprehensive analysis of both with clear justification
- 4 points: Good analysis with minor gaps
- 3 points: Adequate analysis with some missing details
- 2 points: Basic analysis with significant gaps
- 1 point: Incomplete or incorrect

### Question 1.3: Feature Engineering (5 points)

**Expected Answer:**

1. **Entropy:** Measures randomness/compression. Malware often has high entropy due to encryption/packing.
2. **String Count:** Benign files typically have more readable strings; malware may be obfuscated.
3. **Suspicious APIs:** Direct indicator of malicious behavior (e.g., CreateRemoteThread, WriteProcessMemory).
4. **Packer Signature:** Presence of known packers often indicates malware attempting to evade detection.
5. **Digital Signature:** Legitimate files typically have valid signatures; malware rarely does.

**Grading Rubric:**
- 5 points: Five relevant features with clear explanations
- 4 points: Five features with minor explanation gaps
- 3 points: Four-five features with adequate explanations
- 2 points: Three features or weak explanations
- 1 point: Incomplete or incorrect

### Question 1.4: Model Evaluation Metrics (5 points)

**Expected Answer:**

- **Accuracy:** (TP+TN)/(TP+TN+FP+FN). Overall correctness. Important for balanced datasets.
- **Precision:** TP/(TP+FP). Of positive predictions, how many correct. Important when false positives are costly.
- **Recall:** TP/(TP+FN). Of actual positives, how many found. Important when false negatives are costly.
- **F1-Score:** Harmonic mean of precision and recall. Good for imbalanced datasets.

**Grading Rubric:**
- 5 points: All metrics defined with appropriate context
- 4 points: All metrics defined with minor gaps
- 3 points: Most metrics defined adequately
- 2 points: Some metrics missing or incomplete
- 1 point: Incomplete or incorrect

---

## Part 2: Exploratory Data Analysis - Answer Key (30 points)

### Exercise 2.1: Run EDA Script (5 points)

**Expected Results:**

1. **Total samples:** 100 files
2. **Class distribution:**
   - Benign: 50 files (50%)
   - Malicious: 50 files (50%)
3. **Top correlated features:**
   - Suspicious APIs (highest correlation with malicious label)
   - Entropy (high correlation with malicious)
   - Packer Signature (high correlation with malicious)

**Grading Rubric:**
- 5 points: Correct identification of all statistics
- 4 points: Correct with minor discrepancies
- 3 points: Mostly correct with some errors
- 2 points: Partially correct
- 1 point: Incomplete or incorrect

### Exercise 2.2: Analyze Visualizations (10 points)

**Expected Answers:**

1. **Correlation matrix:** Entropy, suspicious_apis, and packer_signature are highly correlated with label.
2. **Feature distributions:** Entropy shows bimodal distribution; string_count and import_count show right-skewed distributions.
3. **Benign vs Malicious:** Entropy and suspicious_apis show clearest separation.
4. **Class balance:** Dataset is perfectly balanced (50-50), which is ideal for model training. No class weighting needed.

**Grading Rubric:**
- 10 points: Comprehensive analysis of all visualizations
- 8 points: Good analysis with minor gaps
- 6 points: Adequate analysis with some missing details
- 4 points: Basic analysis with significant gaps
- 2 points: Incomplete or incorrect

### Exercise 2.3: Statistical Analysis (15 points)

**Expected Answers:**

1. **Entropy statistics:**
   - Benign: Mean ≈ 4.2, Std Dev ≈ 0.15
   - Malicious: Mean ≈ 7.7, Std Dev ≈ 0.10
   - Clear separation indicates strong predictive power

2. **Outliers:** File ID 35 (media_folder.iso) is an outlier with extremely large file size.

3. **Data quality issues:**
   - One extreme outlier (media_folder.iso)
   - Otherwise clean dataset with no missing values
   - Good feature distributions

4. **Proposed preprocessing:**
   - Remove or handle the extreme outlier
   - Normalize/scale features to similar ranges
   - No missing value imputation needed
   - Feature selection to remove highly correlated features

**Grading Rubric:**
- 15 points: Comprehensive statistical analysis with all components
- 12 points: Good analysis with minor gaps
- 9 points: Adequate analysis with some missing elements
- 6 points: Basic analysis with significant gaps
- 3 points: Incomplete or incorrect

---

## Part 3: Model Building - Answer Key (40 points)

### Exercise 3.1: Run Model Training (10 points)

**Expected Results:**

1. **Best accuracy model:** Neural Network (≈95.8% accuracy)
2. **Training times:**
   - Random Forest: ≈2-3 seconds
   - SVM: ≈1-2 seconds
   - Neural Network: ≈5-10 seconds
3. **Overfitting:** Neural Network shows slight overfitting (training accuracy > test accuracy by ~2-3%)

**Grading Rubric:**
- 10 points: Correct identification of results
- 8 points: Mostly correct with minor discrepancies
- 6 points: Partially correct
- 4 points: Some correct elements
- 2 points: Incomplete or incorrect

### Exercise 3.2: Model Comparison (15 points)

**Expected Results:**

| Metric | Random Forest | SVM | Neural Network |
|--------|---------------|-----|-----------------|
| Accuracy | 0.945 | 0.932 | 0.958 |
| Precision | 0.923 | 0.901 | 0.941 |
| Recall | 0.956 | 0.945 | 0.968 |
| F1-Score | 0.939 | 0.922 | 0.954 |
| ROC-AUC | 0.982 | 0.968 | 0.989 |

**Analysis Questions:**

1. **Best model:** Neural Network - highest accuracy, precision, recall, F1-score, and ROC-AUC
2. **Trade-offs:**
   - Random Forest: Faster, interpretable, but slightly lower performance
   - SVM: Fast training, but lower performance
   - Neural Network: Best performance, but slower and less interpretable
3. **Precision vs Recall:** For forensics, recall is more important (catch all malware) but precision matters too (avoid false positives). Neural Network balances both well.

**Grading Rubric:**
- 15 points: Comprehensive comparison with clear justifications
- 12 points: Good comparison with minor gaps
- 9 points: Adequate comparison with some missing analysis
- 6 points: Basic comparison with significant gaps
- 3 points: Incomplete or incorrect

### Exercise 3.3: Hyperparameter Analysis (15 points)

**Expected Approach:**

**Example - Random Forest Hyperparameter Tuning:**

Original:
- n_estimators=100
- max_depth=15
- min_samples_split=5

Modified:
- n_estimators=200 → Improved accuracy by ~0.5%
- max_depth=10 → Reduced overfitting, accuracy ≈ same
- min_samples_split=10 → Reduced overfitting, accuracy ≈ same

**Grading Rubric:**
- 15 points: Systematic hyperparameter tuning with clear documentation
- 12 points: Good tuning with minor documentation gaps
- 9 points: Adequate tuning with some missing details
- 6 points: Basic tuning with significant gaps
- 3 points: Incomplete or incorrect

---

## Part 4: Forensic Analysis - Answer Key (40 points)

### Exercise 4.1: Run Forensic Analysis (10 points)

**Expected Results:**

1. **Malicious files:** 50 (50% of dataset)
2. **Ensemble accuracy:** ≈96.5%
3. **High confidence predictions:** ≈45-48 files (>0.8 confidence)

**Grading Rubric:**
- 10 points: Correct results from analysis
- 8 points: Mostly correct with minor discrepancies
- 6 points: Partially correct
- 4 points: Some correct elements
- 2 points: Incomplete or incorrect

### Exercise 4.2: Analyze Results (15 points)

**Expected Analysis:**

1. **Top 5 high-confidence malicious files:**
   - ransomware_sample_001.exe (confidence ≈ 0.98)
   - trojan_variant.exe (confidence ≈ 0.97)
   - ransomware_variant2.exe (confidence ≈ 0.96)
   - botnet_client.exe (confidence ≈ 0.95)
   - worm_sample.exe (confidence ≈ 0.94)

2. **Suspicious files:** Typically 2-5 files with confidence 0.4-0.6 (depends on model variations)

3. **Model agreement:** ≈92-95% agreement among all three models

4. **Misclassifications:** Few misclassifications; when they occur, usually on borderline cases

**Grading Rubric:**
- 15 points: Comprehensive analysis of all elements
- 12 points: Good analysis with minor gaps
- 9 points: Adequate analysis with some missing details
- 6 points: Basic analysis with significant gaps
- 3 points: Incomplete or incorrect

### Exercise 4.3: Forensic Report (15 points)

**Expected Report Structure:**

**Executive Summary:** 
"Analysis of 100 forensic artifacts using ensemble AI model detected 50 malicious files with 96.5% accuracy. High-confidence malware includes ransomware, trojans, and worms. Immediate quarantine and investigation recommended."

**Methodology:**
"Three machine learning models (Random Forest, SVM, Neural Network) trained on forensic features including entropy, string count, API signatures, and packer detection. Ensemble voting used for final predictions."

**Key Findings:**
"50 malicious files identified, primarily ransomware (15), trojans (12), and worms (10). 45 files with high confidence (>0.8). 5 suspicious files requiring manual review."

**Confidence Assessment:**
"Ensemble model achieved 96.5% accuracy on test set. High model agreement (92-95%) increases confidence. ROC-AUC of 0.989 indicates excellent discrimination."

**Recommendations:**
"1. Immediately quarantine all high-confidence malicious files. 2. Manually review 5 suspicious files. 3. Perform deeper analysis on malware families. 4. Update detection signatures. 5. Monitor for lateral movement."

**Grading Rubric:**
- 15 points: Professional report with all required sections
- 12 points: Good report with minor gaps
- 9 points: Adequate report with some missing sections
- 6 points: Basic report with significant gaps
- 3 points: Incomplete or incorrect

---

## Part 5: Advanced Applications - Answer Key (30 points)

### Exercise 5.1: Ensemble Methods (10 points)

**Expected Approach:**

**Ensemble Method:** Soft voting (average probabilities)

**Implementation:**
```python
ensemble_pred = (rf_prob + svm_prob + nn_prob) / 3
ensemble_label = 1 if ensemble_pred > 0.5 else 0
```

**Results:**
- Ensemble accuracy: ≈96.5%
- Random Forest: ≈94.5%
- SVM: ≈93.2%
- Neural Network: ≈95.8%

**Advantages:**
- Combines strengths of multiple models
- Reduces overfitting
- More robust predictions
- Better generalization

**Disadvantages:**
- Slower inference (3x computation)
- More complex to maintain
- May not improve if models are correlated

**Grading Rubric:**
- 10 points: Clear ensemble approach with proper implementation
- 8 points: Good approach with minor gaps
- 6 points: Adequate approach with some missing details
- 4 points: Basic approach with significant gaps
- 2 points: Incomplete or incorrect

### Exercise 5.2: Explainable AI (10 points)

**Expected Analysis:**

**Most Important Features:**
1. Suspicious APIs (highest importance)
2. Entropy (high importance)
3. Packer Signature (high importance)

**Example Predictions:**
- File: ransomware_sample_001.exe
  - High entropy (7.8) → malicious indicator
  - High suspicious APIs (8) → malicious indicator
  - Packer present → malicious indicator
  - Prediction: Malicious (confidence 0.98) ✓

**Model Limitations:**
- Cannot explain individual feature contributions
- Black-box nature makes debugging difficult
- May miss novel malware with different characteristics
- Requires labeled training data

**Proposed Improvements:**
- Use LIME for local explanations
- Implement SHAP values for feature importance
- Use decision trees for interpretability
- Add human-in-the-loop validation

**Grading Rubric:**
- 10 points: Comprehensive XAI analysis with examples
- 8 points: Good analysis with minor gaps
- 6 points: Adequate analysis with some missing details
- 4 points: Basic analysis with significant gaps
- 2 points: Incomplete or incorrect

### Exercise 5.3: Advanced Topic (10 points)

**Expected Analysis (varies by topic):**

**Transfer Learning Example:**
- Use pre-trained malware models trained on millions of samples
- Fine-tune on forensic dataset (smaller, domain-specific)
- Advantages: Better performance with less data, faster training
- Disadvantages: May not transfer well if domains differ significantly

**Adversarial Robustness Example:**
- Attackers could modify file features to evade detection
- Add small perturbations to entropy, API counts, etc.
- Defense: Adversarial training, ensemble methods, continuous monitoring

**Real-time Detection Example:**
- Deploy model as API service
- Process files as they arrive
- Trade-off: Speed vs accuracy
- Solution: Use faster models (Random Forest) or GPU acceleration

**Explainable AI Example:**
- Explain predictions to non-technical stakeholders
- Use visualizations and simple language
- Show feature contributions and confidence scores
- Provide clear recommendations

**Grading Rubric:**
- 10 points: Comprehensive analysis of chosen topic
- 8 points: Good analysis with minor gaps
- 6 points: Adequate analysis with some missing details
- 4 points: Basic analysis with significant gaps
- 2 points: Incomplete or incorrect

---

## Grading Summary

| Part | Max Points | Typical Range |
|------|-----------|---------------|
| Part 1: Fundamentals | 20 | 16-20 |
| Part 2: EDA | 30 | 24-30 |
| Part 3: Model Building | 40 | 32-40 |
| Part 4: Forensic Analysis | 40 | 32-40 |
| Part 5: Advanced | 30 | 24-30 |
| **Total** | **200** | **160-200** |

---

## Common Student Errors and Corrections

### Error 1: Confusing Precision and Recall
**Incorrect:** "Precision is how many we found"
**Correct:** "Recall is how many we found; Precision is how many were correct"

### Error 2: Ignoring Class Imbalance
**Incorrect:** "Accuracy is the best metric"
**Correct:** "With imbalanced data, F1-score or ROC-AUC is better"

### Error 3: Not Scaling Features for SVM
**Incorrect:** "SVM works the same with or without scaling"
**Correct:** "SVM requires feature scaling for optimal performance"

### Error 4: Overfitting Not Recognized
**Incorrect:** "High training accuracy means the model is good"
**Correct:** "High training accuracy with low test accuracy indicates overfitting"

### Error 5: Misinterpreting Confidence Scores
**Incorrect:** "0.51 confidence means the prediction is reliable"
**Correct:** "0.51 confidence is near random; need >0.7 for reliable predictions"

---

## Teaching Tips

1. **Emphasize the forensic context:** Help students understand why these techniques matter for real investigations.

2. **Use real examples:** Show examples of actual malware and how the model would classify them.

3. **Discuss limitations:** Be honest about model limitations and when human expertise is still needed.

4. **Encourage experimentation:** Have students modify code and observe effects on performance.

5. **Connect to career:** Discuss job opportunities in AI-based forensics and threat intelligence.

---

**Lab Manual Version:** 1.0  
**Last Updated:** November 2025  
**Prepared by:** INT411 Instructional Team
