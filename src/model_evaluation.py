"""
Model Evaluation Module for Telco Customer Churn Prediction

Comprehensive evaluation metrics and visualization for classification models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, log_loss
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model: Any = None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model for evaluation
        """
        self.model = model
        self.metrics = {}
        self.predictions = None
        self.probabilities = None
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, 
                model: Any = None) -> Dict:
        """
        Perform comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: True labels
            model: Model to evaluate (uses self.model if not provided)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        model = model or self.model
        if model is None:
            raise ValueError("No model provided for evaluation")
        
        logger.info("Evaluating model...")
        
        # Make predictions
        self.predictions = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            self.probabilities = model.predict_proba(X_test)[:, 1]
        else:
            self.probabilities = None
        
        # Calculate metrics
        self.metrics = self._calculate_metrics(y_test)
        
        return self.metrics
    
    def _calculate_metrics(self, y_true: pd.Series) -> Dict:
        """Calculate all evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, self.predictions)
        metrics['precision'] = precision_score(y_true, self.predictions, zero_division=0)
        metrics['recall'] = recall_score(y_true, self.predictions, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, self.predictions, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, self.predictions)
        metrics['confusion_matrix'] = cm
        metrics['true_negatives'] = cm[0, 0]
        metrics['false_positives'] = cm[0, 1]
        metrics['false_negatives'] = cm[1, 0]
        metrics['true_positives'] = cm[1, 1]
        
        # Specificity (True Negative Rate)
        metrics['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        
        # Probability-based metrics
        if self.probabilities is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, self.probabilities)
            metrics['average_precision'] = average_precision_score(y_true, self.probabilities)
            metrics['log_loss'] = log_loss(y_true, self.probabilities)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, self.predictions, output_dict=True
        )
        
        logger.info(f"Evaluation complete - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def get_summary(self) -> pd.DataFrame:
        """Get a summary DataFrame of key metrics."""
        if not self.metrics:
            raise ValueError("No metrics available. Run evaluate() first.")
        
        summary_metrics = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity'],
            'Value': [
                self.metrics['accuracy'],
                self.metrics['precision'],
                self.metrics['recall'],
                self.metrics['f1_score'],
                self.metrics['specificity']
            ]
        }
        
        if 'roc_auc' in self.metrics:
            summary_metrics['Metric'].append('ROC-AUC')
            summary_metrics['Value'].append(self.metrics['roc_auc'])
        
        return pd.DataFrame(summary_metrics)
    
    def print_report(self) -> str:
        """Print a formatted evaluation report."""
        if not self.metrics:
            raise ValueError("No metrics available. Run evaluate() first.")
        
        report = """
================================================================================
                           MODEL EVALUATION REPORT
================================================================================

CLASSIFICATION METRICS
----------------------
  Accuracy:    {accuracy:.4f}
  Precision:   {precision:.4f}
  Recall:      {recall:.4f}
  F1-Score:    {f1_score:.4f}
  Specificity: {specificity:.4f}
""".format(**self.metrics)
        
        if 'roc_auc' in self.metrics:
            report += f"""
PROBABILITY METRICS
-------------------
  ROC-AUC:          {self.metrics['roc_auc']:.4f}
  Avg Precision:    {self.metrics['average_precision']:.4f}
  Log Loss:         {self.metrics['log_loss']:.4f}
"""
        
        report += f"""
CONFUSION MATRIX
----------------
              Predicted
              Neg    Pos
  Actual Neg  {self.metrics['true_negatives']:5d}  {self.metrics['false_positives']:5d}
  Actual Pos  {self.metrics['false_negatives']:5d}  {self.metrics['true_positives']:5d}

================================================================================
"""
        
        print(report)
        return report
    
    def plot_confusion_matrix(self, save_path: str = None) -> plt.Figure:
        """Plot confusion matrix heatmap."""
        if not self.metrics:
            raise ValueError("No metrics available. Run evaluate() first.")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            self.metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'],
            ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")
        
        return fig
    
    def plot_roc_curve(self, y_true: pd.Series, save_path: str = None) -> plt.Figure:
        """Plot ROC curve."""
        if self.probabilities is None:
            raise ValueError("Probabilities not available for ROC curve")
        
        fpr, tpr, thresholds = roc_curve(y_true, self.probabilities)
        roc_auc = self.metrics.get('roc_auc', roc_auc_score(y_true, self.probabilities))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"ROC curve saved to: {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: pd.Series, 
                                   save_path: str = None) -> plt.Figure:
        """Plot Precision-Recall curve."""
        if self.probabilities is None:
            raise ValueError("Probabilities not available for PR curve")
        
        precision, recall, thresholds = precision_recall_curve(y_true, self.probabilities)
        avg_precision = self.metrics.get('average_precision', 
                                        average_precision_score(y_true, self.probabilities))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR Curve (AP = {avg_precision:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"PR curve saved to: {save_path}")
        
        return fig


def compare_models(models: Dict[str, Any], X_test: pd.DataFrame, 
                  y_test: pd.Series) -> pd.DataFrame:
    """
    Compare multiple models.
    
    Args:
        models: Dictionary of {model_name: trained_model}
        X_test: Test features
        y_test: True labels
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for name, model in models.items():
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X_test, y_test)
        
        results.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics.get('roc_auc', None)
        })
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    
    return comparison_df


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("Use ModelEvaluator class for comprehensive model evaluation")
