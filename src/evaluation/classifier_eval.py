import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
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
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.train_classifier import ComplaintClassifier, ComplaintDataset

def load_model_and_tokenizer(model_dir):
    """Load the trained model and tokenizer with multi-output support"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load label mappings
    with open(f"{model_dir}/label_mappings.json", "r") as f:
        mappings = json.load(f)
        area_label_encoder = mappings["area_label_encoder"]
        area_label_decoder = mappings["area_label_decoder"]
        use_multioutput = mappings.get("use_multioutput", False)
        
        priority_label_encoder = None
        priority_label_decoder = None
        if use_multioutput:
            priority_label_encoder = mappings["priority_label_encoder"]
            priority_label_decoder = mappings["priority_label_decoder"]
    
    # Convert string keys back to integers for label_decoder if needed
    if all(k.isdigit() for k in area_label_decoder.keys()):
        area_label_decoder = {int(k): v for k, v in area_label_decoder.items()}
    
    if use_multioutput and priority_label_decoder and all(k.isdigit() for k in priority_label_decoder.keys()):
        priority_label_decoder = {int(k): v for k, v in priority_label_decoder.items()}
    
    # Use the ComplaintClassifier class to load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CRITICAL FIX: Use original model name, not model_dir
    original_model_name = "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis"
    
    if use_multioutput:
        classifier = ComplaintClassifier(
            model_name=original_model_name,  # Use original model name
            num_area_labels=len(area_label_encoder),
            num_priority_labels=len(priority_label_encoder) if priority_label_encoder else 3,
            use_multioutput=True
        )
    else:
        classifier = ComplaintClassifier(
            model_name=original_model_name,  # Use original model name
            num_area_labels=len(area_label_encoder),
            use_multioutput=False
        )
    
    # Load the trained weights
    classifier.load_model(model_dir)
    
    print(f"Model loaded from {model_dir} to {device}")
    print(f"Multi-output: {use_multioutput}")
    if use_multioutput:
        print(f"Area classes: {len(area_label_encoder)}")
        print(f"Priority classes: {len(priority_label_encoder) if priority_label_encoder else 0}")
    
    return classifier

def load_test_data(data_path, text_column, area_label_column, priority_label_column, classifier):
    """Load and preprocess test data with multi-output support"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading test data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Ensure text column exists
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in data file")
    
    # Ensure area label column exists
    if area_label_column not in df.columns:
        raise ValueError(f"Area label column '{area_label_column}' not found in data file")
    
    # Check priority label column for multi-output
    has_priority = priority_label_column and priority_label_column in df.columns
    if classifier.use_multioutput and not has_priority:
        print(f"Warning: Priority label column '{priority_label_column}' not found. Evaluation will be area-only.")
    
    # Clean data
    df[text_column] = df[text_column].fillna("")
    df[text_column] = df[text_column].apply(classifier._clean_text)
    
    # Convert area labels to numerical values
    df['area_label_id'] = df[area_label_column].map(classifier.area_label_encoder)
    
    # Convert priority labels if available
    priority_labels = None
    if has_priority and classifier.use_multioutput:
        df['priority_label_id'] = df[priority_label_column].map(classifier.priority_label_encoder)
        priority_labels = df['priority_label_id'].tolist()
    
    # Create dataset
    test_dataset = ComplaintDataset(
        texts=df[text_column].tolist(),
        area_labels=df['area_label_id'].tolist(),
        priority_labels=priority_labels,
        tokenizer=classifier.tokenizer,
        max_length=classifier.max_length,
        use_multioutput=classifier.use_multioutput
    )
    
    print(f"Test data loaded with {len(df)} samples")
    
    return test_dataset, df

def evaluate_model(classifier, test_dataset, batch_size=16):
    """Evaluate model performance with multiple metrics for both area and priority"""
    device = classifier.device
    model = classifier.model
    tokenizer = classifier.tokenizer
    area_label_decoder = classifier.area_label_decoder
    priority_label_decoder = getattr(classifier, 'priority_label_decoder', None)
    use_multioutput = classifier.use_multioutput
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Collect all predictions and true labels
    all_area_labels = []
    all_area_preds = []
    all_area_probs = []
    
    all_priority_labels = []
    all_priority_preds = []
    all_priority_probs = []
    
    # Process batches
    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get model outputs
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            if use_multioutput:
                # Multi-output model
                area_logits = outputs['area_logits']
                priority_logits = outputs['priority_logits']
                
                # Area predictions
                area_probs = torch.nn.functional.softmax(area_logits, dim=1)
                area_preds = torch.argmax(area_probs, dim=1).cpu().numpy()
                
                # Priority predictions
                priority_probs = torch.nn.functional.softmax(priority_logits, dim=1)
                priority_preds = torch.argmax(priority_probs, dim=1).cpu().numpy()
                
                # Store results
                all_area_labels.extend(batch['area_labels'].cpu().numpy())
                all_area_preds.extend(area_preds)
                all_area_probs.extend(area_probs.cpu().numpy())
                
                all_priority_labels.extend(batch['priority_labels'].cpu().numpy())
                all_priority_preds.extend(priority_preds)
                all_priority_probs.extend(priority_probs.cpu().numpy())
            else:
                # Single output model (area only)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                
                # Store results
                all_area_labels.extend(batch['labels'].cpu().numpy())
                all_area_preds.extend(preds)
                all_area_probs.extend(probs.cpu().numpy())
    
    # Convert to arrays for easier processing
    y_area_true = np.array(all_area_labels)
    y_area_pred = np.array(all_area_preds)
    y_area_probs = np.array(all_area_probs)
    
    # Ensure label decoders have integer keys
    area_label_decoder = {int(k): v for k, v in area_label_decoder.items()}
    
    # Get human-readable labels for area
    y_area_true_labels = [area_label_decoder[int(i)] for i in y_area_true]
    y_area_pred_labels = [area_label_decoder[int(i)] for i in y_area_pred]
    
    # Calculate area metrics
    area_metrics = calculate_classification_metrics(
        y_area_true, y_area_pred, y_area_probs, area_label_decoder, "AREA"
    )
    
    # Print area classification report
    print("\nAREA Classification Report:")
    print(classification_report(y_area_true_labels, y_area_pred_labels))
    
    # Calculate priority metrics if multi-output
    priority_metrics = None
    if use_multioutput and len(all_priority_labels) > 0:
        y_priority_true = np.array(all_priority_labels)
        y_priority_pred = np.array(all_priority_preds)
        y_priority_probs = np.array(all_priority_probs)
        
        priority_label_decoder = {int(k): v for k, v in priority_label_decoder.items()}
        
        # Get human-readable labels for priority
        y_priority_true_labels = [priority_label_decoder[int(i)] for i in y_priority_true]
        y_priority_pred_labels = [priority_label_decoder[int(i)] for i in y_priority_pred]
        
        priority_metrics = calculate_classification_metrics(
            y_priority_true, y_priority_pred, y_priority_probs, priority_label_decoder, "PRIORITY"
        )
        
        # Print priority classification report
        print("\nPRIORITY Classification Report:")
        print(classification_report(y_priority_true_labels, y_priority_pred_labels))
    
    # Plot results (including confusion matrices)
    plot_classification_results(area_metrics, priority_metrics, use_multioutput)
    plot_confusion_matrices(area_metrics, priority_metrics, use_multioutput)
    
    # Return all metrics
    results = {
        'area_metrics': area_metrics,
        'use_multioutput': use_multioutput
    }
    
    if priority_metrics:
        results['priority_metrics'] = priority_metrics
        results['y_priority_true'] = y_priority_true
        results['y_priority_pred'] = y_priority_pred
        results['y_priority_probs'] = y_priority_probs
    
    results.update({
        'y_area_true': y_area_true,
        'y_area_pred': y_area_pred,
        'y_area_probs': y_area_probs
    })
    
    return results

def calculate_classification_metrics(y_true, y_pred, y_probs, label_decoder, task_name):
    """Calculate classification metrics for a single task"""
    class_metrics = {}
    num_classes = len(label_decoder)
    
    # Binary classification metrics for each class (one-vs-rest)
    for class_idx in range(num_classes):
        class_name = label_decoder[class_idx]
        
        # Create binary labels for this class
        binary_true = (y_true == class_idx).astype(int)
        binary_pred = (y_pred == class_idx).astype(int)
        
        # Class-specific probabilities
        class_probs = y_probs[:, class_idx]
        
        # Calculate binary metrics
        precision = precision_score(binary_true, binary_pred, zero_division=0)
        recall = recall_score(binary_true, binary_pred, zero_division=0)
        f1 = f1_score(binary_true, binary_pred, zero_division=0)
        
        # Calculate PR and ROC curves
        precisions, recalls, pr_thresholds = precision_recall_curve(binary_true, class_probs)
        try:
            pr_auc = auc(recalls, precisions)
        except:
            pr_auc = 0.0
            
        fpr, tpr, roc_thresholds = roc_curve(binary_true, class_probs)
        try:
            roc_auc = auc(fpr, tpr)
        except:
            roc_auc = 0.0
        
        # Store metrics
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'pr_curve': (precisions, recalls),
            'roc_curve': (fpr, tpr)
        }
    
    # Print confusion matrix
    print(f"\n{task_name} Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Calculate weighted average metrics
    class_counts = np.bincount(y_true, minlength=num_classes)
    weights = class_counts / len(y_true)
    
    weighted_precision = sum(class_metrics[label_decoder[i]]['precision'] * weights[i] for i in range(num_classes))
    weighted_recall = sum(class_metrics[label_decoder[i]]['recall'] * weights[i] for i in range(num_classes))
    weighted_f1 = sum(class_metrics[label_decoder[i]]['f1'] * weights[i] for i in range(num_classes))
    weighted_pr_auc = sum(class_metrics[label_decoder[i]]['pr_auc'] * weights[i] for i in range(num_classes))
    weighted_roc_auc = sum(class_metrics[label_decoder[i]]['roc_auc'] * weights[i] for i in range(num_classes))
    
    # Print weighted metrics
    print(f"\n{task_name} Weighted Average Metrics:")
    print(f"Precision: {weighted_precision:.4f}")
    print(f"Recall: {weighted_recall:.4f}")
    print(f"F1 Score: {weighted_f1:.4f}")
    print(f"PR AUC: {weighted_pr_auc:.4f}")
    print(f"ROC AUC: {weighted_roc_auc:.4f}")
    
    return {
        'class_metrics': class_metrics,
        'weighted_metrics': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1,
            'pr_auc': weighted_pr_auc,
            'roc_auc': weighted_roc_auc
        },
        'confusion_matrix': cm,
        'label_decoder': label_decoder,
        'y_true': y_true,
        'y_pred': y_pred
    }

def plot_confusion_matrices(area_metrics, priority_metrics=None, use_multioutput=False):
    """Plot confusion matrices for area and optionally priority classification"""
    
    if use_multioutput and priority_metrics:
        # Create subplots for both area and priority confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Area confusion matrix
        plot_single_confusion_matrix(area_metrics, axes[0], "AREA")
        
        # Priority confusion matrix
        plot_single_confusion_matrix(priority_metrics, axes[1], "PRIORITY")
        
        # Save the plot
        plot_file = 'multioutput_confusion_matrices.png'
    else:
        # Single output (area only)
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Area confusion matrix
        plot_single_confusion_matrix(area_metrics, ax, "AREA")
        
        # Save the plot
        plot_file = 'area_confusion_matrix.png'
    
    plt.tight_layout()
    
    file_path = '../../data/models/classifier/results'
    os.makedirs(file_path, exist_ok=True)
    plt.savefig(os.path.join(file_path, plot_file), dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrices saved to {os.path.join(file_path, plot_file)}")

def plot_single_confusion_matrix(metrics, ax, task_name):
    """Plot confusion matrix for a single task"""
    cm = metrics['confusion_matrix']
    label_decoder = metrics['label_decoder']
    
    # Get class names in order
    class_names = [label_decoder[i] for i in range(len(label_decoder))]
    
    # Create a normalized version for percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot using seaborn heatmap
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    # Customize the plot
    ax.set_title(f'{task_name} Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Add total counts as text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            if count > 0:
                # Add count in smaller text below the percentage
                text = ax.texts[i * len(class_names) + j]
                current_text = text.get_text()
                text.set_text(f'{current_text}\n({count})')
                text.set_fontsize(9)

def plot_classification_results(area_metrics, priority_metrics=None, use_multioutput=False):
    """Plot PR and ROC curves for area and optionally priority classification"""
    
    if use_multioutput and priority_metrics:
        # Create subplots for both area and priority
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Area plots
        plot_task_curves(area_metrics, axes[0], "AREA", max_classes=5)
        
        # Priority plots  
        plot_task_curves(priority_metrics, axes[1], "PRIORITY", max_classes=3)
        
        # Save the plot
        plot_file = 'multioutput_model_performance.png'
    else:
        # Single output (area only)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Area plots
        plot_task_curves(area_metrics, axes, "AREA", max_classes=5)
        
        # Save the plot
        plot_file = 'area_model_performance.png'
    
    plt.tight_layout()
    
    file_path = '../../data/models/classifier/results'
    os.makedirs(file_path, exist_ok=True)
    plt.savefig(os.path.join(file_path, plot_file), dpi=300, bbox_inches='tight')
    print(f"\nPerformance curves saved to {os.path.join(file_path, plot_file)}")

def plot_task_curves(metrics, axes, task_name, max_classes=5):
    """Plot PR and ROC curves for a specific task"""
    class_metrics = metrics['class_metrics']
    label_decoder = metrics['label_decoder']
    
    # Limit to top classes for readability
    num_classes = min(len(label_decoder), max_classes)
    
    # PR curves
    ax_pr = axes[0]
    for class_idx in range(num_classes):
        class_name = label_decoder[class_idx]
        if class_name in class_metrics:
            precisions, recalls = class_metrics[class_name]['pr_curve']
            pr_auc = class_metrics[class_name]['pr_auc']
            ax_pr.plot(recalls, precisions, label=f"{class_name} (AUC = {pr_auc:.3f})")
    
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title(f'{task_name} Precision-Recall Curves (top {num_classes} classes)')
    ax_pr.grid(True)
    ax_pr.legend(loc='lower left')
    
    # ROC curves
    ax_roc = axes[1]
    for class_idx in range(num_classes):
        class_name = label_decoder[class_idx]
        if class_name in class_metrics:
            fpr, tpr = class_metrics[class_name]['roc_curve']
            roc_auc = class_metrics[class_name]['roc_auc']
            ax_roc.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})")
    
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f'{task_name} ROC Curves (top {num_classes} classes)')
    ax_roc.grid(True)
    ax_roc.legend(loc='lower right')

def plot_class_distribution(results, use_multioutput=False):
    """Plot the distribution of classes in true vs predicted labels"""
    
    if use_multioutput:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Area distribution
        plot_single_distribution(
            results['y_area_true'], 
            results['y_area_pred'], 
            results['area_metrics']['label_decoder'],
            axes[0], 
            "AREA"
        )
        
        # Priority distribution
        if 'y_priority_true' in results:
            plot_single_distribution(
                results['y_priority_true'],
                results['y_priority_pred'],
                results['priority_metrics']['label_decoder'],
                axes[1],
                "PRIORITY"
            )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        plot_single_distribution(
            results['y_area_true'],
            results['y_area_pred'],
            results['area_metrics']['label_decoder'],
            ax,
            "AREA"
        )
    
    plt.tight_layout()
    
    # Save the plot
    dist_file = 'multioutput_class_distribution.png' if use_multioutput else 'area_class_distribution.png'
    file_path = '../../data/models/classifier/results'
    plt.savefig(os.path.join(file_path, dist_file), dpi=300, bbox_inches='tight')
    print(f"Class distribution plot saved to {os.path.join(file_path, dist_file)}")

def plot_single_distribution(y_true, y_pred, label_decoder, ax, task_name):
    """Plot distribution for a single task"""
    num_classes = len(label_decoder)
    
    # Count occurrences of each class
    true_counts = np.bincount(y_true, minlength=num_classes)
    pred_counts = np.bincount(y_pred, minlength=num_classes)
    
    # Get class names
    class_names = [label_decoder[int(i)] for i in range(num_classes)]
    
    # Sort classes by true count for better visualization
    sorted_idx = np.argsort(-true_counts)
    top_classes = sorted_idx[:min(10, num_classes)]  # Show top 10 classes
    
    x = np.arange(len(top_classes))
    width = 0.35
    
    # Plot bar chart
    ax.bar(x - width/2, true_counts[top_classes], width, label='True', alpha=0.8)
    ax.bar(x + width/2, pred_counts[top_classes], width, label='Predicted', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(f'{task_name} Class Distribution: True vs Predicted')
    ax.set_xticks(x)
    ax.set_xticklabels([class_names[i] for i in top_classes], rotation=45, ha='right')
    ax.legend()

def analyze_misclassifications(test_df, results, use_multioutput=False, top_n=5):
    """Analyze the most severe misclassifications for both tasks"""
    
    print(f"\n{'='*60}")
    print("MISCLASSIFICATION ANALYSIS")
    print(f"{'='*60}")
    
    # Area misclassification analysis
    analyze_single_task_misclassifications(
        test_df, 
        results['y_area_true'],
        results['y_area_pred'],
        results['y_area_probs'],
        results['area_metrics']['label_decoder'],
        "AREA",
        top_n
    )
    
    # Priority misclassification analysis
    if use_multioutput and 'y_priority_true' in results:
        print(f"\n{'-'*40}")
        analyze_single_task_misclassifications(
            test_df,
            results['y_priority_true'],
            results['y_priority_pred'],
            results['y_priority_probs'],
            results['priority_metrics']['label_decoder'],
            "PRIORITY",
            top_n
        )

def analyze_single_task_misclassifications(test_df, y_true, y_pred, y_probs, label_decoder, task_name, top_n=5):
    """Analyze misclassifications for a single task"""
    
    # Create a DataFrame with predictions
    results_df = test_df.copy()
    results_df[f'{task_name.lower()}_true_label_id'] = y_true
    results_df[f'{task_name.lower()}_pred_label_id'] = y_pred
    results_df[f'{task_name.lower()}_confidence'] = np.max(y_probs, axis=1)
    
    # Add human-readable labels
    results_df[f'{task_name.lower()}_true_label'] = results_df[f'{task_name.lower()}_true_label_id'].map(label_decoder)
    results_df[f'{task_name.lower()}_pred_label'] = results_df[f'{task_name.lower()}_pred_label_id'].map(label_decoder)
    
    # Identify misclassifications
    results_df[f'{task_name.lower()}_is_correct'] = (results_df[f'{task_name.lower()}_true_label_id'] == results_df[f'{task_name.lower()}_pred_label_id'])
    misclassified = results_df[~results_df[f'{task_name.lower()}_is_correct']]
    
    print(f"\n{task_name} Misclassification Analysis:")
    print(f"Total misclassifications: {len(misclassified)} out of {len(results_df)} ({len(misclassified)/len(results_df)*100:.2f}%)")
    
    # Find most common error types (true -> predicted)
    error_types = misclassified.groupby([f'{task_name.lower()}_true_label', f'{task_name.lower()}_pred_label']).size().reset_index(name='count')
    error_types = error_types.sort_values('count', ascending=False)
    
    print(f"\nTop 10 Most Common {task_name} Error Types:")
    for i, row in error_types.head(10).iterrows():
        print(f"{row[f'{task_name.lower()}_true_label']} -> {row[f'{task_name.lower()}_pred_label']}: {row['count']} instances")
    
    # Find high-confidence errors
    high_conf_errors = misclassified.sort_values(f'{task_name.lower()}_confidence', ascending=False)
    
    print(f"\nTop {top_n} Highest Confidence {task_name} Errors:")
    text_column = 'observaciones' if 'observaciones' in test_df.columns else 'text'
    for i, row in high_conf_errors.head(top_n).iterrows():
        print(f"\nConfidence: {row[f'{task_name.lower()}_confidence']:.3f}")
        print(f"True: {row[f'{task_name.lower()}_true_label']} | Predicted: {row[f'{task_name.lower()}_pred_label']}")
        if text_column in row:
            print(f"Text: {row[text_column][:100]}...")
    
    return misclassified, error_types

def save_detailed_results(test_df, results, use_multioutput=False, output_file='multioutput_detailed_results.csv'):
    """Save detailed prediction results to CSV file"""
    # Create a DataFrame with all predictions
    results_df = test_df.copy()
    
    # Area predictions
    results_df['area_true_label_id'] = results['y_area_true']
    results_df['area_pred_label_id'] = results['y_area_pred']
    results_df['area_confidence'] = np.max(results['y_area_probs'], axis=1)
    
    area_label_decoder = results['area_metrics']['label_decoder']
    results_df['area_true_label'] = results_df['area_true_label_id'].map(area_label_decoder)
    results_df['area_pred_label'] = results_df['area_pred_label_id'].map(area_label_decoder)
    results_df['area_is_correct'] = (results_df['area_true_label_id'] == results_df['area_pred_label_id'])
    
    # Add area probabilities for each class
    for i in range(results['y_area_probs'].shape[1]):
        class_name = area_label_decoder[i]
        results_df[f'area_prob_{class_name}'] = results['y_area_probs'][:, i]
    
    # Priority predictions (if multi-output)
    if use_multioutput and 'y_priority_true' in results:
        results_df['priority_true_label_id'] = results['y_priority_true']
        results_df['priority_pred_label_id'] = results['y_priority_pred']
        results_df['priority_confidence'] = np.max(results['y_priority_probs'], axis=1)
        
        priority_label_decoder = results['priority_metrics']['label_decoder']
        results_df['priority_true_label'] = results_df['priority_true_label_id'].map(priority_label_decoder)
        results_df['priority_pred_label'] = results_df['priority_pred_label_id'].map(priority_label_decoder)
        results_df['priority_is_correct'] = (results_df['priority_true_label_id'] == results_df['priority_pred_label_id'])
        
        # Add priority probabilities for each class
        for i in range(results['y_priority_probs'].shape[1]):
            class_name = priority_label_decoder[i]
            results_df[f'priority_prob_{class_name}'] = results['y_priority_probs'][:, i]
        
        # Combined correctness (both area and priority correct)
        results_df['both_correct'] = results_df['area_is_correct'] & results_df['priority_is_correct']
    
    # Save to CSV
    file_path = '../../data/models/classifier/results'
    os.makedirs(file_path, exist_ok=True)
    results_df.to_csv(os.path.join(file_path, output_file), index=False) 
    print(f"Detailed results saved to {os.path.join(file_path, output_file)}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Evaluate Multi-output Transformer-based Complaint Classifier')
    parser.add_argument('--model', help='Path to the trained model directory', default='../../data/models/classifier/classifier_model')
    parser.add_argument('--data', help='Path to test data CSV file', default='../../data/processed/test/test_classifier.csv')
    parser.add_argument('--text-col', help='Name of column containing text data', default='observaciones')
    parser.add_argument('--area-col', help='Name of column containing area label data', default='areaServicioDescripcion')
    parser.add_argument('--priority-col', help='Name of column containing priority label data', default='prioridadDescripcion')
    parser.add_argument('--batch-size', help='Batch size for evaluation', type=int, default=16)
    parser.add_argument('--analyze-errors', help='Analyze misclassifications', action='store_true')
    parser.add_argument('--output', help='Output file for detailed results', default='multioutput_detailed_results.csv')
    
    args = parser.parse_args()
    
    print("🔍 Loading model and preparing evaluation...")
    
    # Load model
    classifier = load_model_and_tokenizer(args.model)
    
    # Load test data
    test_dataset, test_df = load_test_data(
        args.data, 
        args.text_col, 
        args.area_col,
        args.priority_col,
        classifier
    )
    
    # Evaluate model
    print("\n🚀 Starting model evaluation...")
    results = evaluate_model(classifier, test_dataset, batch_size=args.batch_size)
    
    # Plot class distribution
    print("\n📊 Generating visualizations...")
    plot_class_distribution(results, results['use_multioutput'])
    
    # Analyze errors if requested
    if args.analyze_errors:
        print("\n🔍 Analyzing misclassifications...")
        analyze_misclassifications(
            test_df,
            results,
            results['use_multioutput']
        )
    
    # Save detailed results
    print("\n💾 Saving detailed results...")
    save_detailed_results(
        test_df,
        results,
        results['use_multioutput'],
        args.output
    )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    if results['use_multioutput']:
        print("✅ Multi-output model evaluated for AREA and PRIORITY classification")
        if 'priority_metrics' in results:
            area_f1 = results['area_metrics']['weighted_metrics']['f1']
            priority_f1 = results['priority_metrics']['weighted_metrics']['f1']
            print(f"📊 Area F1-Score: {area_f1:.4f}")
            print(f"📊 Priority F1-Score: {priority_f1:.4f}")
            
            # Combined accuracy (both area and priority correct)
            if 'y_priority_true' in results:
                area_correct = (results['y_area_true'] == results['y_area_pred'])
                priority_correct = (results['y_priority_true'] == results['y_priority_pred'])
                both_correct = area_correct & priority_correct
                combined_accuracy = np.mean(both_correct)
                print(f"🎯 Combined Accuracy (both correct): {combined_accuracy:.4f}")
        else:
            area_f1 = results['area_metrics']['weighted_metrics']['f1']
            print(f"📊 Area F1-Score: {area_f1:.4f}")
            print("⚠️  Priority evaluation skipped (no priority labels in test data)")
    else:
        print("✅ Single-output model evaluated for AREA classification only")
        area_f1 = results['area_metrics']['weighted_metrics']['f1']
        print(f"📊 Area F1-Score: {area_f1:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()