"""
Evaluation metrics for the multi-stream transformer model.

This module provides various metrics for evaluating model performance
on event sequence prediction tasks.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate accuracy for categorical predictions.
    
    Args:
        predictions: Predicted class indices
        targets: Target class indices
        
    Returns:
        Accuracy score
    """
    pred_classes = torch.argmax(predictions, dim=-1)
    correct = (pred_classes == targets).float()
    return correct.mean().item()

def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity score
    """
    return np.exp(loss)

def calculate_sequence_accuracy(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    exact_match: bool = False
) -> float:
    """
    Calculate sequence-level accuracy.
    
    Args:
        predictions: Predicted sequences of shape (seq_len, vocab_size)
        targets: Target sequences of shape (seq_len,)
        exact_match: If True, requires exact sequence match
        
    Returns:
        Sequence accuracy score
    """
    pred_classes = torch.argmax(predictions, dim=-1)
    
    if exact_match:
        # Exact sequence match
        return (pred_classes == targets).all().float().item()
    else:
        # Token-level accuracy
        return (pred_classes == targets).float().mean().item()

def calculate_category_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for each category separately.
    
    Args:
        predictions: Dictionary of predictions per category
        targets: Dictionary of targets per category
        
    Returns:
        Dictionary of metrics per category
    """
    metrics = {}
    
    for category in predictions.keys():
        if category in targets:
            pred = predictions[category]
            tgt = targets[category]
            
            # Calculate accuracy
            accuracy = calculate_accuracy(pred, tgt)
            
            # Calculate perplexity (assuming cross-entropy loss)
            loss = torch.nn.functional.cross_entropy(pred, tgt)
            perplexity = calculate_perplexity(loss.item())
            
            metrics[category] = {
                'accuracy': accuracy,
                'perplexity': perplexity,
                'loss': loss.item()
            }
    
    return metrics

def calculate_bleu_score(predictions: List[List[int]], targets: List[List[int]]) -> float:
    """
    Calculate BLEU score for sequence generation (simplified version).
    
    Args:
        predictions: List of predicted sequences
        targets: List of target sequences
        
    Returns:
        BLEU score
    """
    # Simplified BLEU calculation
    # In practice, you'd use nltk.translate.bleu_score
    
    total_score = 0.0
    for pred, tgt in zip(predictions, targets):
        # Calculate n-gram overlap (simplified)
        pred_set = set(pred)
        tgt_set = set(tgt)
        
        if len(tgt_set) > 0:
            overlap = len(pred_set.intersection(tgt_set))
            score = overlap / len(tgt_set)
            total_score += score
    
    return total_score / len(predictions) if predictions else 0.0

class MetricsTracker:
    """
    Class to track and compute metrics during training.
    """
    
    def __init__(self, category_names: List[str]):
        self.category_names = category_names
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.losses = []
        self.accuracies = []
        self.category_losses = {name: [] for name in self.category_names}
        self.category_accuracies = {name: [] for name in self.category_names}
    
    def update(
        self, 
        total_loss: float, 
        per_field_losses: List[float],
        predictions: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Update metrics with new batch results.
        
        Args:
            total_loss: Total loss for the batch
            per_field_losses: List of losses per category
            predictions: Model predictions
            targets: Target values
        """
        self.losses.append(total_loss)
        
        # Calculate overall accuracy
        if predictions is not None and targets is not None:
            accuracy = calculate_accuracy(predictions, targets)
            self.accuracies.append(accuracy)
        
        # Update per-category metrics
        for i, (name, loss) in enumerate(zip(self.category_names, per_field_losses)):
            self.category_losses[name].append(loss)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics across all tracked batches."""
        metrics = {}
        
        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)
            metrics['std_loss'] = np.std(self.losses)
        
        if self.accuracies:
            metrics['avg_accuracy'] = np.mean(self.accuracies)
            metrics['std_accuracy'] = np.std(self.accuracies)
        
        # Per-category metrics
        for name in self.category_names:
            if self.category_losses[name]:
                metrics[f'{name}_loss'] = np.mean(self.category_losses[name])
        
        return metrics
    
    def print_summary(self):
        """Print a summary of tracked metrics."""
        metrics = self.get_average_metrics()
        
        print("Metrics Summary:")
        print(f"  Average Loss: {metrics.get('avg_loss', 0):.4f} ± {metrics.get('std_loss', 0):.4f}")
        
        if 'avg_accuracy' in metrics:
            print(f"  Average Accuracy: {metrics['avg_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
        
        print("  Per-Category Losses:")
        for name in self.category_names:
            loss_key = f'{name}_loss'
            if loss_key in metrics:
                print(f"    {name}: {metrics[loss_key]:.4f}")