"""
Multi-Stream Transformer Training Pipeline

This script implements a comprehensive training pipeline for the multi-stream
transformer architecture, including:
- Data loading and preprocessing
- Model training with validation
- TensorBoard logging and monitoring
- Model checkpointing and resumption
- Performance metrics tracking

Author: [Your Name]
Date: 2024
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.events_loader import EventsLoader
from modeling.models.model import ModelTrainer

def load_config(path: str) -> Dict[str, Any]:
    """Load training configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration to JSON file."""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)

def collate_fn(batch):
    """Custom collate function for batching samples."""
    # batch: list of (tgt, src, [mask])
    tgts = [item[0] for item in batch]
    srcs = [item[1] for item in batch]
    masks = [item[2] for item in batch]
    return srcs, tgts, masks

def setup_logging(run_name: str) -> tuple:
    """Setup TensorBoard logging directories."""
    log_dir = Path("runs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    train_writer = SummaryWriter(log_dir / "train")
    val_writer = SummaryWriter(log_dir / "val")
    
    return train_writer, val_writer, log_dir

def execute_epoch(
    model_trainer: ModelTrainer,
    dataloader: DataLoader,
    writer: SummaryWriter,
    epoch: int,
    grad_accum: int,
    is_training: bool = True,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Execute one epoch of training or validation.
    
    Args:
        model_trainer: The model trainer instance
        dataloader: Data loader for the epoch
        writer: TensorBoard writer for logging
        epoch: Current epoch number
        grad_accum: Gradient accumulation steps
        is_training: Whether this is a training epoch
        device: Device to run on
        
    Returns:
        Dictionary of epoch metrics
    """
    model_trainer.model.train() if is_training else model_trainer.model.eval()
    
    total_loss = 0.0
    total_samples = 0
    step_times = []
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"{'Train' if is_training else 'Val'} Epoch {epoch}")
    
    for step, (batch_src, batch_tgt, batch_masks) in enumerate(pbar):
        start_time = time.time()
        
        # Move data to device
        batch_src = [
            {k: v.to(device, non_blocking=True) for k, v in src.items()}
            for src in batch_src
        ]
        batch_tgt = [t.to(device, non_blocking=True) for t in batch_tgt]
        batch_masks = [
            [m.to(device, non_blocking=True) for m in masks]
            for masks in batch_masks
        ]
        
        # Forward pass
        with torch.set_grad_enabled(is_training):
            loss, per_field_losses = model_trainer(
                batch_src, batch_tgt, batch_masks, run_backward=is_training
            )
        
        # Update metrics
        batch_size = len(batch_src)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Optimizer step for training
        if is_training:
            if ((step + 1) % grad_accum == 0) or (step + 1 == len(dataloader)):
                if model_trainer.optimizer is not None:
                    model_trainer.optimizer.step()
                    model_trainer.optimizer.zero_grad()
        
        # Logging
        global_step = epoch * len(dataloader) + step
        writer.add_scalar("Loss/Total", loss.item(), global_step)
        
        # Log per-field losses
        for name, field_loss in zip(model_trainer.category_names, per_field_losses):
            writer.add_scalar(f"Loss/{name}", field_loss.item(), global_step)
        
        # Timing
        step_time = time.time() - start_time
        step_times.append(step_time)
        writer.add_scalar("Time/Step", step_time, global_step)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/total_samples:.4f}',
            'time': f'{step_time:.2f}s'
        })
    
    # Calculate epoch metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_time = sum(step_times) / len(step_times) if step_times else 0.0
    
    metrics = {
        'avg_loss': avg_loss,
        'total_samples': total_samples,
        'avg_step_time': avg_time,
        'total_time': sum(step_times)
    }
    
    return metrics

def save_checkpoint(
    model_trainer: ModelTrainer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: Path,
    is_best: bool = False
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_trainer.model.state_dict(),
        'optimizer_state_dict': model_trainer.optimizer.state_dict() if model_trainer.optimizer else None,
        'metrics': metrics,
        'model_info': model_trainer.get_model_info()
    }
    
    torch.save(checkpoint, save_path / f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, save_path / "checkpoint_latest.pt")
    
    if is_best:
        torch.save(checkpoint, save_path / "checkpoint_best.pt")

def load_checkpoint(model_trainer: ModelTrainer, checkpoint_path: str) -> int:
    """Load model checkpoint and return the epoch number."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model_trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    if model_trainer.optimizer and 'optimizer_state_dict' in checkpoint:
        model_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Multi-Stream Transformer')
    parser.add_argument('--config', type=str, default='modeling/training_config.json',
                       help='Path to training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on (cuda/cpu)')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this training run')
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    base_dir = Path(__file__).parent.parent  # Go up one level from modeling/
    config_path = base_dir / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(str(config_path))
    
    print("Training Configuration:")
    print(json.dumps(cfg, indent=2))
    
    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create run name
    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Run name: {run_name}")
    
    # Setup logging
    train_writer, val_writer, log_dir = setup_logging(run_name)
    
    # Save configuration
    save_config(cfg, log_dir / "config.json")
    
    # Initialize dataset
    print("Loading dataset...")
    dataset = EventsLoader(
        batch_size=cfg['batch_size'],
        src_seq_length=cfg['src_seq_len'],
        tgt_seq_length=cfg['tgt_seq_len'],
        id_category_size=cfg['id_category_size']
    )
    dataset.load_dataset()
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Category fields: {list(dataset.category_fields.keys())}")
    
    # Split dataset
    n_total = len(dataset)
    n_train = int(0.8 * n_total)  # 80% train
    n_val = int(0.1 * n_total)    # 10% validation
    n_test = n_total - n_train - n_val  # 10% test
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test]
    )
    
    print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Initialize model
    print("Initializing model...")
    model_trainer = ModelTrainer(
        d_input=dataset.input_size,
        d_model=cfg['model']['d_model'],
        n_heads=cfg['model']['num_heads'],
        encoder_layers=cfg['model']['encoder_layers'],
        decoder_layers=cfg['model']['decoder_layers'],
        d_categories=list(dataset.category_fields.values()),
        encoders=dataset.encoder_streams,
        d_output=dataset.output_size,
        dropout=cfg['model'].get('dropout', 0.1)
    ).to(device)
    
    # Print model info
    model_info = model_trainer.get_model_info()
    print("Model Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Setup optimizer
    model_trainer.optimizer = torch.optim.AdamW(
        model_trainer.model.parameters(),
        lr=cfg.get('learning_rate', 3e-4),
        weight_decay=cfg.get('weight_decay', 0.01)
    )
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_trainer.optimizer,
        T_max=cfg['epochs'],
        eta_min=cfg.get('min_lr', 1e-6)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(model_trainer, args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(start_epoch, cfg['epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")
        
        # Training phase
        train_metrics = execute_epoch(
            model_trainer, train_loader, train_writer, epoch,
            cfg['grad_accum'], is_training=True, device=device
        )
        
        # Validation phase
        val_metrics = execute_epoch(
            model_trainer, val_loader, val_writer, epoch,
            cfg['grad_accum'], is_training=False, device=device
        )
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        train_writer.add_scalar("Learning_Rate", current_lr, epoch)
        
        # Log epoch metrics
        print(f"Train Loss: {train_metrics['avg_loss']:.4f}")
        print(f"Val Loss: {val_metrics['avg_loss']:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Save checkpoint
        is_best = val_metrics['avg_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['avg_loss']
            print(f"New best validation loss: {best_val_loss:.4f}")
        
        save_checkpoint(
            model_trainer, epoch, 
            {'train': train_metrics, 'val': val_metrics},
            log_dir, is_best
        )
        
        # Early stopping check (optional)
        if cfg.get('early_stopping_patience'):
            # Implementation left as exercise
            pass
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Logs and checkpoints saved to: {log_dir}")
    
    # Close writers
    train_writer.close()
    val_writer.close()

if __name__ == "__main__":
    main()
