#!/usr/bin/env python3
"""
INT411 AI for Forensics - Part 3 & 4: Model Training and Evaluation
This script builds, trains, and evaluates multiple ML models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ForensicModelTrainer:
    """Class for training and evaluating forensic AI models"""
    
    def __init__(self, dataset_path):
        """Initialize trainer with dataset"""
        self.dataset_path = dataset_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare dataset"""
        print("[*] Loading dataset...")
        self.df = pd.read_csv(self.dataset_path)
        print(f"[+] Dataset loaded: {self.df.shape}")
        
        # Separate features and labels
        X = self.df.drop(['file_id', 'file_name', 'file_type', 'label'], axis=1)
        y = self.df['label']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"[+] Training set: {self.X_train.shape}")
        print(f"[+] Test set: {self.X_test.shape}")
        print(f"[+] Features: {list(X.columns)}")
        
        return X, y
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*60)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Train model
        print("[*] Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, self.y_train)
        print("[+] Random Forest training complete")
        
        # Evaluate
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        results = self._evaluate_model(y_pred, y_pred_proba, 'Random Forest')
        self.models['Random Forest'] = (rf_model, scaler)
        self.results['Random Forest'] = results
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Important Features:")
        print(feature_importance.head())
        
        return rf_model, scaler, feature_importance
    
    def train_svm(self):
        """Train Support Vector Machine model"""
        print("\n" + "="*60)
        print("TRAINING SUPPORT VECTOR MACHINE (SVM)")
        print("="*60)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Train model
        print("[*] Training SVM...")
        svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        svm_model.fit(X_train_scaled, self.y_train)
        print("[+] SVM training complete")
        
        # Evaluate
        y_pred = svm_model.predict(X_test_scaled)
        y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]
        
        results = self._evaluate_model(y_pred, y_pred_proba, 'SVM')
        self.models['SVM'] = (svm_model, scaler)
        self.results['SVM'] = results
        
        return svm_model, scaler
    
    def train_neural_network(self):
        """Train Neural Network model"""
        print("\n" + "="*60)
        print("TRAINING NEURAL NETWORK")
        print("="*60)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Build model
        print("[*] Building neural network...")
        nn_model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        nn_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        print("[*] Training neural network...")
        history = nn_model.fit(
            X_train_scaled, self.y_train,
            epochs=50,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )
        print("[+] Neural network training complete")
        
        # Evaluate
        y_pred_proba = nn_model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        results = self._evaluate_model(y_pred, y_pred_proba, 'Neural Network')
        self.models['Neural Network'] = (nn_model, scaler)
        self.results['Neural Network'] = results
        
        return nn_model, scaler, history
    
    def _evaluate_model(self, y_pred, y_pred_proba, model_name):
        """Evaluate model performance"""
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n{model_name} Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Benign', 'Malicious']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_df = pd.DataFrame(self.results).T
        print("\nModel Performance Comparison:")
        print(comparison_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']])
        
        # Visualize comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Metrics comparison
        metrics_df = comparison_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
        metrics_df.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Model Performance Metrics Comparison')
        axes[0].set_ylabel('Score')
        axes[0].set_xlabel('Model')
        axes[0].legend(loc='best')
        axes[0].set_ylim([0.8, 1.0])
        
        # ROC curves
        for model_name in self.results:
            y_pred_proba = self.results[model_name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = self.results[model_name]['roc_auc']
            axes[1].plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})')
        
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curves Comparison')
        axes[1].legend(loc='best')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n[+] Model comparison saved to model_comparison.png")
        plt.close()
    
    def save_models(self, output_dir):
        """Save trained models"""
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, (model, scaler) in self.models.items():
            model_file = output_path / f"{model_name.lower().replace(' ', '_')}_model.pkl"
            scaler_file = output_path / f"{model_name.lower().replace(' ', '_')}_scaler.pkl"
            
            joblib.dump(model, model_file)
            joblib.dump(scaler, scaler_file)
            print(f"[+] Saved {model_name} model and scaler")
    
    def main(self):
        """Run complete training pipeline"""
        print("\n" + "="*60)
        print("INT411 AI FOR FORENSICS - MODEL TRAINING")
        print("="*60)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Train models
        self.train_random_forest()
        self.train_svm()
        self.train_neural_network()
        
        # Compare models
        self.compare_models()
        
        # Save models
        self.save_models('models')
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)

def main():
    """Main entry point"""
    dataset_path = Path(__file__).parent.parent / 'datasets' / 'forensic_dataset.csv'
    trainer = ForensicModelTrainer(dataset_path)
    trainer.main()

if __name__ == "__main__":
    main()
