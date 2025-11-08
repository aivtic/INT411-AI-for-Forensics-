#!/usr/bin/env python3
"""
INT411 AI for Forensics - Part 2: Exploratory Data Analysis
This script performs comprehensive EDA on the forensic dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_dataset(filepath):
    """Load forensic dataset from CSV"""
    print("[*] Loading forensic dataset...")
    df = pd.read_csv(filepath)
    print(f"[+] Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    return df

def basic_statistics(df):
    """Display basic dataset statistics"""
    print("\n" + "="*60)
    print("BASIC DATASET STATISTICS")
    print("="*60)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    
    # Class distribution
    print(f"\nClass Distribution:")
    print(f"  Benign (0): {(df['label'] == 0).sum()} samples ({(df['label'] == 0).sum()/len(df)*100:.1f}%)")
    print(f"  Malicious (1): {(df['label'] == 1).sum()} samples ({(df['label'] == 1).sum()/len(df)*100:.1f}%)")

def feature_analysis(df):
    """Analyze individual features"""
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)
    
    # Numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns
    
    for feature in numerical_features:
        if feature == 'label':
            continue
        print(f"\n{feature}:")
        print(f"  Mean: {df[feature].mean():.2f}")
        print(f"  Median: {df[feature].median():.2f}")
        print(f"  Std Dev: {df[feature].std():.2f}")
        print(f"  Min: {df[feature].min():.2f}")
        print(f"  Max: {df[feature].max():.2f}")

def correlation_analysis(df):
    """Analyze feature correlations"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Select numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlation with label
    correlations = df[numerical_features].corr()['label'].sort_values(ascending=False)
    print("\nFeature Correlation with Label:")
    print(correlations)
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_features].corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("\n[+] Correlation matrix saved to correlation_matrix.png")
    plt.close()

def distribution_analysis(df):
    """Analyze feature distributions"""
    print("\n" + "="*60)
    print("DISTRIBUTION ANALYSIS")
    print("="*60)
    
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('label')
    
    # Create subplots for distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, feature in enumerate(numerical_features):
        axes[idx].hist(df[feature], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {feature}')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
    
    # Remove extra subplots
    for idx in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    print("[+] Feature distributions saved to feature_distributions.png")
    plt.close()

def malicious_vs_benign(df):
    """Compare malicious vs benign samples"""
    print("\n" + "="*60)
    print("MALICIOUS VS BENIGN ANALYSIS")
    print("="*60)
    
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('label')
    
    # Create comparison plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, feature in enumerate(numerical_features):
        benign = df[df['label'] == 0][feature]
        malicious = df[df['label'] == 1][feature]
        
        axes[idx].hist(benign, bins=20, alpha=0.6, label='Benign', edgecolor='black')
        axes[idx].hist(malicious, bins=20, alpha=0.6, label='Malicious', edgecolor='black')
        axes[idx].set_title(f'{feature}: Benign vs Malicious')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
    
    # Remove extra subplots
    for idx in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('benign_vs_malicious.png', dpi=300, bbox_inches='tight')
    print("[+] Benign vs Malicious comparison saved to benign_vs_malicious.png")
    plt.close()

def class_balance_analysis(df):
    """Analyze class balance"""
    print("\n" + "="*60)
    print("CLASS BALANCE ANALYSIS")
    print("="*60)
    
    class_counts = df['label'].value_counts()
    
    plt.figure(figsize=(8, 6))
    class_counts.plot(kind='bar', color=['green', 'red'], edgecolor='black')
    plt.title('Class Distribution')
    plt.xlabel('Class (0=Benign, 1=Malicious)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print("[+] Class distribution saved to class_distribution.png")
    plt.close()

def main():
    """Main analysis function"""
    print("\n" + "="*60)
    print("INT411 AI FOR FORENSICS - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load dataset
    dataset_path = Path(__file__).parent.parent / 'datasets' / 'forensic_dataset.csv'
    df = load_dataset(dataset_path)
    
    # Perform analyses
    basic_statistics(df)
    feature_analysis(df)
    correlation_analysis(df)
    distribution_analysis(df)
    malicious_vs_benign(df)
    class_balance_analysis(df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated visualizations:")
    print("  - correlation_matrix.png")
    print("  - feature_distributions.png")
    print("  - benign_vs_malicious.png")
    print("  - class_distribution.png")

if __name__ == "__main__":
    main()
