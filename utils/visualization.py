"""
Visualization utilities for model training and analysis.

This module provides functions for visualizing training metrics,
model architecture, and attention patterns.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    source_tokens: List[str],
    target_tokens: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights tensor of shape (tgt_len, src_len)
        source_tokens: List of source token names
        target_tokens: List of target token names
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Convert to numpy if tensor
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    sns.heatmap(
        attention_weights,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        cmap='Blues',
        cbar=True,
        square=True
    )
    
    plt.title('Attention Weights Heatmap', fontsize=16)
    plt.xlabel('Source Tokens', fontsize=12)
    plt.ylabel('Target Tokens', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_metrics(
    metrics_dict: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple metrics on the same figure.
    
    Args:
        metrics_dict: Dictionary mapping metric names to lists of values
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for metric_name, values in metrics_dict.items():
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, label=metric_name, linewidth=2)
    
    plt.title('Training Metrics', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_sequence_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    category_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize sequence predictions vs targets.
    
    Args:
        predictions: Predicted sequences
        targets: Target sequences
        category_names: Names of the categories
        save_path: Optional path to save the plot
    """
    n_categories = min(len(category_names), 6)  # Limit to 6 for readability
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(n_categories):
        ax = axes[i]
        
        # Plot predictions and targets
        seq_len = min(50, predictions.shape[0])  # Limit sequence length for visibility
        x = range(seq_len)
        
        ax.plot(x, predictions[:seq_len, i].cpu().numpy(), 'b-', label='Predicted', alpha=0.7)
        ax.plot(x, targets[:seq_len, i].cpu().numpy(), 'r--', label='Target', alpha=0.7)
        
        ax.set_title(f'{category_names[i]}', fontsize=12)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()