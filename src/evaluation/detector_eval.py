import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from sklearn.metrics import (
    precision_recall_curve, 
    roc_curve, 
    auc, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
import argparse
import os
import dill as pickle

def load_model(model_path):
    """Load model and handle both old and new formats"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Check if it's the new format with metadata
    if isinstance(model_data, dict):
        print("Loading model with metadata...")
        if 'model' in model_data:
            print(f"Model trained on: {model_data.get('training_date', 'Unknown')}")
            print(f"Training F1 score: {model_data.get('test_f1_score', 'Unknown')}")
            print(f"Training samples: {model_data.get('training_samples', 'Unknown')}")
            if model_data.get('best_params'):
                print(f"Optimized hyperparameters: {model_data['best_params']}")
            return model_data['model']
        else:
            raise ValueError("Dictionary format found but no 'model' key present")
    else:
        # Old format - just the model object
        print("Loading legacy model format...")
        return model_data

def load_data(labeled_data_path):
    """Load labeled data for evaluation"""
    if not os.path.exists(labeled_data_path):
        raise FileNotFoundError(f"Labeled data file not found: {labeled_data_path}")
    
    data = pd.read_csv(labeled_data_path)
    
    # Verify required columns
    required_cols = ['text', 'is_valid_complaint']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Labeled data must have columns: {required_cols}")
    
    # Drop rows with missing values
    data = data.dropna(subset=required_cols)
    data = data[data['text'].duplicated()==False] #drop duplicados de texto
    
    # Convert target to int for evaluation
    data['is_valid_complaint'] = data['is_valid_complaint'].astype(int)
    
    print(f"Loaded {len(data)} labeled examples")
    return data

def evaluate_model(model, X, y):
    """Evaluate model performance with multiple metrics"""
    # Get predictions and probabilities
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Calculate basic metrics
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Get PR curve
    precisions, recalls, pr_thresholds = precision_recall_curve(y, y_prob)
    pr_auc = auc(recalls, precisions)
    
    # Get ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    # Print summary metrics
    print(f"\nSummary Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Plot PR curve
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(recalls, precisions, 'b-', label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend(loc='lower left')
    
    # Plot ROC curve
    plt.subplot(2, 1, 2)
    plt.plot(fpr, tpr, 'r-', label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = 'model_performance_curves.png'
    file_path = '../../data/models/detector/results'
    plt.savefig(os.path.join(file_path, plot_file))
    print(f"\nPerformance curves saved to {os.path.join(file_path, plot_file)}")
    
    # Optionally display the plot
    plt.show()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'y_prob': y_prob,
        'y_pred': y_pred
    }

def find_optimal_threshold(y_true, y_prob):
    """Find the threshold that maximizes F1 score"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Calculate F1 score for each threshold
    f1_scores = []
    for i in range(len(thresholds)):
        # Convert probabilities to predictions using current threshold
        y_pred = (y_prob >= thresholds[i]).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    # Find threshold with best F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"\nOptimal Threshold Analysis:")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Best F1 score: {best_f1:.4f}")
    
    # Plot thresholds vs F1 scores
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores[:-1], 'g-')
    plt.axvline(x=best_threshold, color='r', linestyle='--')
    plt.xlabel('Probability Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Probability Threshold')
    plt.grid(True)
    
    # Save the plot
    threshold_file = 'optimal_threshold.png'
    plt.savefig(threshold_file)
    print(f"Optimal threshold plot saved to {threshold_file}")
    
    # Optionally display the plot
    plt.show()
    
    return best_threshold, best_f1

def analyze_misclassifications(X, y_true, y_pred, y_prob, top_n=10):
    """Analyze the most severe misclassifications"""
    # Create a DataFrame with all predictions
    results_df = pd.DataFrame({
        'text': X,
        'actual': y_true,
        'predicted': y_pred,
        'probability': y_prob
    })
    
    # Find false positives
    fp = results_df[(results_df['actual'] == 0) & (results_df['predicted'] == 1)]
    fp_sorted = fp.sort_values('probability', ascending=False)
    
    # Find false negatives
    fn = results_df[(results_df['actual'] == 1) & (results_df['predicted'] == 0)]
    fn_sorted = fn.sort_values('probability', ascending=True)
    
    print(f"\nMisclassification Analysis:")
    print(f"False Positives: {len(fp)} cases")
    print(f"False Negatives: {len(fn)} cases")
    
    # Top false positives (most confident mistakes)
    if len(fp) > 0:
        print(f"\nTop {min(top_n, len(fp))} False Positives (Non-complaints classified as complaints):")
        for i, (_, row) in enumerate(fp_sorted.head(top_n).iterrows()):
            print(f"\n{i+1}. Probability: {row['probability']:.3f}")
            print(f"Text: {row['text'][:200]}...")
    
    # Top false negatives (most confident mistakes in other direction)
    if len(fn) > 0:
        print(f"\nTop {min(top_n, len(fn))} False Negatives (Valid complaints missed):")
        for i, (_, row) in enumerate(fn_sorted.head(top_n).iterrows()):
            print(f"\n{i+1}. Probability: {row['probability']:.3f}")
            print(f"Text: {row['text'][:200]}...")
    
    return fp_sorted, fn_sorted

def main():
    parser = argparse.ArgumentParser(description='Evaluate Complaint Classifier Model')
    parser.add_argument('--model', help='Path to the trained model file', default='../../data/models/detector/detector_model/complaint_detector_model.pkl')
    parser.add_argument('--data', help='Path to labeled data for evaluation', default='../../data/processed/test/test_detector.csv')
    parser.add_argument('--threshold', help='Custom threshold for classification', type=float, default=0.5)
    parser.add_argument('--find-threshold', help='Find optimal threshold for F1 score', action='store_true')
    parser.add_argument('--analyze-errors', help='Analyze misclassifications', action='store_true')
    
    args = parser.parse_args()
    
    if not args.data:
        parser.error("The --data argument is required")
    
    # Load model and data
    model = load_model(args.model)
    data = load_data(args.data)
    
    # Extract features and target
    X = data['text']
    y = data['is_valid_complaint']
    
    # Evaluate model with default threshold
    print(f"\nEvaluating model with threshold = {args.threshold}")
    results = evaluate_model(model, X, y)
    
    # Find optimal threshold if requested
    if args.find_threshold:
        best_threshold, _ = find_optimal_threshold(y, results['y_prob'])
        
        # Re-evaluate with best threshold
        print(f"\nRe-evaluating model with optimal threshold = {best_threshold:.4f}")
        y_pred_optimal = (results['y_prob'] >= best_threshold).astype(int)
        
        # Print new performance metrics
        precision = precision_score(y, y_pred_optimal)
        recall = recall_score(y, y_pred_optimal)
        f1 = f1_score(y, y_pred_optimal)
        
        print("\nPerformance with optimal threshold:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report with optimal threshold:")
        print(classification_report(y, y_pred_optimal))
    
    # Analyze misclassifications if requested
    if args.analyze_errors:
        analyze_misclassifications(X, y, results['y_pred'], results['y_prob'])

if __name__ == "__main__":
    main()