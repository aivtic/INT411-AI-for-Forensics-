#!/usr/bin/env python3
"""
INT411 AI for Forensics - Part 5: Forensic Analysis with AI
This script applies trained models to analyze new forensic data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from datetime import datetime
import json

class ForensicAnalyzer:
    """Class for analyzing forensic data with trained AI models"""
    
    def __init__(self, model_dir='models', dataset_path=None):
        """Initialize analyzer with trained models"""
        self.model_dir = Path(model_dir)
        self.dataset_path = dataset_path
        self.models = {}
        self.scalers = {}
        self.load_models()
        
    def load_models(self):
        """Load trained models and scalers"""
        print("[*] Loading trained models...")
        
        model_files = list(self.model_dir.glob('*_model.pkl'))
        
        for model_file in model_files:
            model_name = model_file.stem.replace('_model', '').replace('_', ' ').title()
            scaler_file = self.model_dir / f"{model_file.stem.replace('_model', '')}_scaler.pkl"
            
            try:
                self.models[model_name] = joblib.load(model_file)
                self.scalers[model_name] = joblib.load(scaler_file)
                print(f"[+] Loaded {model_name} model")
            except Exception as e:
                print(f"[-] Error loading {model_name}: {e}")
    
    def analyze_file(self, file_data):
        """Analyze a single file using all models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            scaler = self.scalers[model_name]
            
            # Prepare features
            features = file_data[['entropy', 'string_count', 'section_count', 
                                 'import_count', 'suspicious_apis', 
                                 'packer_signature', 'digital_signature']].values.reshape(1, -1)
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features_scaled)[0][1]
            else:
                probability = model.predict(features_scaled)[0][0]
            
            prediction = 1 if probability > 0.5 else 0
            
            predictions[model_name] = {
                'prediction': prediction,
                'confidence': probability,
                'label': 'Malicious' if prediction == 1 else 'Benign'
            }
        
        return predictions
    
    def analyze_dataset(self, dataset_path=None):
        """Analyze entire dataset"""
        if dataset_path is None:
            dataset_path = self.dataset_path
        
        print(f"[*] Analyzing dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        results = []
        
        for idx, row in df.iterrows():
            file_predictions = self.analyze_file(row)
            
            result = {
                'file_id': row['file_id'],
                'file_name': row['file_name'],
                'file_type': row['file_type'],
                'true_label': row['label'],
                'true_label_name': 'Malicious' if row['label'] == 1 else 'Benign'
            }
            
            # Add model predictions
            for model_name, pred in file_predictions.items():
                result[f'{model_name}_prediction'] = pred['prediction']
                result[f'{model_name}_confidence'] = pred['confidence']
            
            # Ensemble prediction (majority vote)
            predictions = [pred['prediction'] for pred in file_predictions.values()]
            ensemble_pred = 1 if sum(predictions) > len(predictions) / 2 else 0
            result['ensemble_prediction'] = ensemble_pred
            result['ensemble_label'] = 'Malicious' if ensemble_pred == 1 else 'Benign'
            
            # Average confidence
            confidences = [pred['confidence'] for pred in file_predictions.values()]
            result['ensemble_confidence'] = np.mean(confidences)
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def generate_report(self, results_df, output_file='forensic_analysis_report.txt'):
        """Generate forensic analysis report"""
        print(f"\n[*] Generating forensic analysis report...")
        
        report = []
        report.append("="*70)
        report.append("INT411 AI FOR FORENSICS - FORENSIC ANALYSIS REPORT")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Files Analyzed: {len(results_df)}")
        
        # Summary statistics
        report.append("\n" + "-"*70)
        report.append("SUMMARY STATISTICS")
        report.append("-"*70)
        
        malicious_count = (results_df['ensemble_prediction'] == 1).sum()
        benign_count = (results_df['ensemble_prediction'] == 0).sum()
        
        report.append(f"\nEnsemble Predictions:")
        report.append(f"  Malicious: {malicious_count} ({malicious_count/len(results_df)*100:.1f}%)")
        report.append(f"  Benign: {benign_count} ({benign_count/len(results_df)*100:.1f}%)")
        
        # Model accuracy
        report.append("\n" + "-"*70)
        report.append("MODEL ACCURACY")
        report.append("-"*70)
        
        for model_name in self.models.keys():
            pred_col = f'{model_name}_prediction'
            accuracy = (results_df[pred_col] == results_df['true_label']).sum() / len(results_df)
            report.append(f"\n{model_name}:")
            report.append(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Ensemble accuracy
        ensemble_accuracy = (results_df['ensemble_prediction'] == results_df['true_label']).sum() / len(results_df)
        report.append(f"\nEnsemble:")
        report.append(f"  Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
        
        # High confidence predictions
        report.append("\n" + "-"*70)
        report.append("HIGH CONFIDENCE MALICIOUS FILES")
        report.append("-"*70)
        
        high_conf_malicious = results_df[
            (results_df['ensemble_prediction'] == 1) & 
            (results_df['ensemble_confidence'] > 0.8)
        ].sort_values('ensemble_confidence', ascending=False)
        
        if len(high_conf_malicious) > 0:
            report.append(f"\nFound {len(high_conf_malicious)} high-confidence malicious files:")
            for idx, row in high_conf_malicious.head(10).iterrows():
                report.append(f"\n  File: {row['file_name']}")
                report.append(f"    Confidence: {row['ensemble_confidence']:.4f}")
                report.append(f"    Type: {row['file_type']}")
        else:
            report.append("\nNo high-confidence malicious files detected.")
        
        # Suspicious files (low confidence)
        report.append("\n" + "-"*70)
        report.append("SUSPICIOUS FILES (LOW CONFIDENCE)")
        report.append("-"*70)
        
        suspicious = results_df[
            (results_df['ensemble_confidence'] > 0.4) & 
            (results_df['ensemble_confidence'] < 0.6)
        ].sort_values('ensemble_confidence')
        
        if len(suspicious) > 0:
            report.append(f"\nFound {len(suspicious)} suspicious files requiring manual review:")
            for idx, row in suspicious.head(10).iterrows():
                report.append(f"\n  File: {row['file_name']}")
                report.append(f"    Confidence: {row['ensemble_confidence']:.4f}")
                report.append(f"    Type: {row['file_type']}")
        else:
            report.append("\nNo suspicious files detected.")
        
        # Recommendations
        report.append("\n" + "-"*70)
        report.append("RECOMMENDATIONS")
        report.append("-"*70)
        report.append("\n1. Quarantine all high-confidence malicious files immediately")
        report.append("2. Manually review suspicious files (confidence 0.4-0.6)")
        report.append("3. Perform deeper analysis on detected malware families")
        report.append("4. Update detection signatures based on findings")
        report.append("5. Monitor for lateral movement and data exfiltration")
        
        report.append("\n" + "="*70)
        
        # Write report
        report_text = "\n".join(report)
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"[+] Report saved to {output_file}")
        return report_text
    
    def visualize_results(self, results_df, output_file='analysis_visualization.png'):
        """Visualize analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Prediction distribution
        pred_counts = results_df['ensemble_prediction'].value_counts()
        axes[0, 0].bar(['Benign', 'Malicious'], 
                       [pred_counts.get(0, 0), pred_counts.get(1, 0)],
                       color=['green', 'red'], edgecolor='black')
        axes[0, 0].set_title('Ensemble Predictions Distribution')
        axes[0, 0].set_ylabel('Count')
        
        # Confidence distribution
        axes[0, 1].hist(results_df['ensemble_confidence'], bins=30, edgecolor='black')
        axes[0, 1].set_title('Confidence Score Distribution')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')
        
        # Confidence by prediction
        benign_conf = results_df[results_df['ensemble_prediction'] == 0]['ensemble_confidence']
        malicious_conf = results_df[results_df['ensemble_prediction'] == 1]['ensemble_confidence']
        
        axes[1, 0].hist(benign_conf, bins=20, alpha=0.6, label='Benign', edgecolor='black')
        axes[1, 0].hist(malicious_conf, bins=20, alpha=0.6, label='Malicious', edgecolor='black')
        axes[1, 0].set_title('Confidence by Prediction')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Model agreement
        model_names = [m for m in self.models.keys()]
        agreement_data = []
        for idx, row in results_df.iterrows():
            predictions = [row[f'{m}_prediction'] for m in model_names]
            agreement = sum(predictions) / len(predictions)
            agreement_data.append(agreement)
        
        axes[1, 1].hist(agreement_data, bins=20, edgecolor='black')
        axes[1, 1].set_title('Model Agreement Distribution')
        axes[1, 1].set_xlabel('Agreement Ratio')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[+] Visualization saved to {output_file}")
        plt.close()

def main():
    """Main analysis function"""
    print("\n" + "="*70)
    print("INT411 AI FOR FORENSICS - FORENSIC ANALYSIS")
    print("="*70)
    
    # Initialize analyzer
    dataset_path = Path(__file__).parent.parent / 'datasets' / 'forensic_dataset.csv'
    analyzer = ForensicAnalyzer(model_dir='models', dataset_path=dataset_path)
    
    # Analyze dataset
    results_df = analyzer.analyze_dataset()
    
    # Save results
    results_df.to_csv('analysis_results.csv', index=False)
    print("[+] Results saved to analysis_results.csv")
    
    # Generate report
    report = analyzer.generate_report(results_df)
    print("\n" + report)
    
    # Visualize results
    analyzer.visualize_results(results_df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
