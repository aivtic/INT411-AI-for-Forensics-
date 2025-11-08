# INT411 AI for Forensics - Student Worksheet

**Name:** ________________________  
**Student ID:** ________________________  
**Date:** ________________________  
**Total Points:** 200

---

## Part 1: AI and Forensics Fundamentals (20 points)

### Question 1.1: Machine Learning Basics (5 points)
Explain the difference between supervised learning and unsupervised learning in the context of digital forensics. Provide two examples of each.

**Your Answer:**
```
Supervised Learning:
- Definition: 
- Example 1: 
- Example 2: 

Unsupervised Learning:
- Definition: 
- Example 1: 
- Example 2: 
```

### Question 1.2: Classification Task (5 points)
In this lab, we're building a binary classification model to distinguish between benign and malicious files. What are the consequences of:
- **False Positives** (benign files classified as malicious):
- **False Negatives** (malicious files classified as benign):

Which is more problematic in forensics and why?

**Your Answer:**
```
False Positives Consequences:

False Negatives Consequences:

More Problematic:
```

### Question 1.3: Feature Engineering (5 points)
List 5 features that would be useful for distinguishing malicious from benign files. For each, explain why it's predictive.

**Your Answer:**
```
Feature 1: 
  Why predictive: 

Feature 2: 
  Why predictive: 

Feature 3: 
  Why predictive: 

Feature 4: 
  Why predictive: 

Feature 5: 
  Why predictive: 
```

### Question 1.4: Model Evaluation Metrics (5 points)
Define the following metrics and explain when each is most important:
- **Accuracy:**
- **Precision:**
- **Recall:**
- **F1-Score:**

**Your Answer:**
```
Accuracy:
  Definition: 
  When important: 

Precision:
  Definition: 
  When important: 

Recall:
  Definition: 
  When important: 

F1-Score:
  Definition: 
  When important: 
```

---

## Part 2: Exploratory Data Analysis (30 points)

### Exercise 2.1: Run EDA Script (5 points)
Run the exploratory analysis script and document the findings.

```bash
python code/01_exploratory_analysis.py
```

**Questions:**
1. How many samples are in the dataset?
2. What is the class distribution (benign vs malicious)?
3. Which features have the highest correlation with the label?

**Your Answers:**
```
1. Total samples: 

2. Class distribution:
   - Benign: 
   - Malicious: 

3. Top correlated features:
   - Feature 1: 
   - Feature 2: 
   - Feature 3: 
```

### Exercise 2.2: Analyze Visualizations (10 points)
Examine the generated visualizations and answer the following:

1. **correlation_matrix.png:** Which features are most correlated with each other?
2. **feature_distributions.png:** Which features have skewed distributions?
3. **benign_vs_malicious.png:** Which features show the clearest separation between classes?
4. **class_distribution.png:** Is the dataset balanced? How might this affect model training?

**Your Answers:**
```
1. Highly correlated features:

2. Skewed distributions:

3. Features with clear separation:

4. Class balance analysis:
```

### Exercise 2.3: Statistical Analysis (15 points)
Based on the EDA results, answer the following:

1. Calculate the mean and standard deviation for entropy in benign vs malicious files.
2. Identify any potential outliers or anomalies.
3. Discuss any data quality issues discovered.
4. Propose preprocessing steps needed before model training.

**Your Answers:**
```
1. Entropy statistics:
   Benign - Mean: , Std Dev: 
   Malicious - Mean: , Std Dev: 

2. Outliers/Anomalies:

3. Data quality issues:

4. Proposed preprocessing:
```

---

## Part 3: Model Building (40 points)

### Exercise 3.1: Run Model Training (10 points)
Run the model training script and document the results.

```bash
python code/02_model_training.py
```

**Questions:**
1. Which model achieved the highest accuracy?
2. What were the training times for each model?
3. Did any model show signs of overfitting?

**Your Answers:**
```
1. Best accuracy model: 
   Accuracy: 

2. Training times:
   - Random Forest: 
   - SVM: 
   - Neural Network: 

3. Overfitting analysis:
```

### Exercise 3.2: Model Comparison (15 points)
Compare the three models based on the generated comparison visualization.

| Metric | Random Forest | SVM | Neural Network |
|--------|---------------|-----|-----------------|
| Accuracy | | | |
| Precision | | | |
| Recall | | | |
| F1-Score | | | |
| ROC-AUC | | | |

**Analysis Questions:**
1. Which model is best for this forensic task and why?
2. What are the trade-offs between the models?
3. How would you choose between precision and recall for this application?

**Your Answers:**
```
1. Best model: 
   Justification: 

2. Trade-offs:

3. Precision vs Recall choice:
```

### Exercise 3.3: Hyperparameter Analysis (15 points)
Modify the model training script to experiment with different hyperparameters for one model.

1. Choose one model (Random Forest, SVM, or Neural Network)
2. Modify 2-3 hyperparameters
3. Document the changes and their effects on performance

**Your Experiment:**
```
Model chosen: 

Hyperparameter 1: 
  Original value: 
  New value: 
  Effect on performance: 

Hyperparameter 2: 
  Original value: 
  New value: 
  Effect on performance: 

Hyperparameter 3: 
  Original value: 
  New value: 
  Effect on performance: 

Conclusion:
```

---

## Part 4: Forensic Analysis (40 points)

### Exercise 4.1: Run Forensic Analysis (10 points)
Run the forensic analysis script to analyze the dataset with trained models.

```bash
python code/03_forensic_analysis.py
```

**Questions:**
1. How many files were classified as malicious?
2. What was the ensemble model accuracy?
3. How many files had high confidence predictions (>0.8)?

**Your Answers:**
```
1. Malicious files: 

2. Ensemble accuracy: 

3. High confidence predictions: 
```

### Exercise 4.2: Analyze Results (15 points)
Examine the analysis results and generated report.

1. **High-Confidence Malicious Files:** List the top 5 files flagged as malicious with high confidence.
2. **Suspicious Files:** Identify files with low confidence (0.4-0.6) that require manual review.
3. **Model Agreement:** How often did all three models agree on predictions?
4. **False Positives/Negatives:** Identify any misclassifications and analyze why they occurred.

**Your Analysis:**
```
1. Top 5 high-confidence malicious files:
   - 
   - 
   - 
   - 
   - 

2. Suspicious files requiring review:

3. Model agreement analysis:

4. Misclassification analysis:
```

### Exercise 4.3: Forensic Report (15 points)
Based on the analysis results, write a professional forensic report including:

1. **Executive Summary** (3-4 sentences)
2. **Methodology** (explain the AI approach)
3. **Key Findings** (what was detected)
4. **Confidence Assessment** (how confident are the results)
5. **Recommendations** (next steps for investigation)

**Your Report:**
```
FORENSIC ANALYSIS REPORT

Executive Summary:


Methodology:


Key Findings:


Confidence Assessment:


Recommendations:
```

---

## Part 5: Advanced Applications (30 points)

### Exercise 5.1: Ensemble Methods (10 points)
Implement an ensemble method combining predictions from all three models.

1. Describe your ensemble approach (voting, averaging, stacking, etc.)
2. Calculate the ensemble accuracy
3. Compare with individual model accuracies
4. Discuss advantages and disadvantages

**Your Work:**
```
Ensemble approach:

Implementation:

Results:
  - Ensemble accuracy: 
  - Random Forest accuracy: 
  - SVM accuracy: 
  - Neural Network accuracy: 

Advantages:

Disadvantages:
```

### Exercise 5.2: Explainable AI (10 points)
Analyze model predictions using explainability techniques.

1. Identify the most important features for predictions
2. Explain why certain files were classified as malicious
3. Discuss limitations of the models
4. Propose improvements for interpretability

**Your Analysis:**
```
Most important features:

Example predictions explanation:
  File 1: 
  File 2: 
  File 3: 

Model limitations:

Proposed improvements:
```

### Exercise 5.3: Advanced Topic (10 points)
Choose one advanced topic and provide a detailed analysis:

**Options:**
- Transfer Learning: How could pre-trained models improve forensic analysis?
- Adversarial Robustness: How could attackers evade the AI model?
- Real-time Detection: How would you deploy this model in production?
- Explainable AI: How would you explain model decisions to non-technical stakeholders?

**Your Analysis:**
```
Topic chosen: 

Detailed analysis:


Recommendations:
```

---

## Submission Checklist

- [ ] All questions answered completely
- [ ] Code runs without errors
- [ ] Visualizations generated and analyzed
- [ ] Report is professional and well-organized
- [ ] All files saved and organized
- [ ] Submitted by deadline

---

**Total Points Earned:** _____ / 200

**Instructor Comments:**

---

**Submission Date:** ________________________

**Instructor Signature:** ________________________
