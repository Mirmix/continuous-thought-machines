"""
Training script to compare baseline CTM with neuron-level normalized CTM on maze-solving task.

This script evaluates the impact of biologically-inspired neuron-level normalization
on training stability, biological plausibility, and interpretability of neural synchrony.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import os
import json
import argparse
from datetime import datetime
import logging

# Add the parent directory to the path to import models
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.ctm_normalized import create_ctm_with_normalization, create_baseline_ctm
from data.custom_datasets import MazeImageFolder


def setup_logging(log_dir):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_model(model_type, config, device):
    """Create either baseline or normalized CTM model."""
    if model_type == 'normalized':
        model = create_ctm_with_normalization(config)
        logging.info("Created normalized CTM model")
    else:
        model = create_baseline_ctm(config)
        logging.info("Created baseline CTM model")
    
    model = model.to(device)
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, model_type):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions, certainties, _ = model(data)
        
        # Use the last prediction for loss computation
        final_predictions = predictions[:, :, -1]  # Shape: (B, out_dims)
        
        # For maze task, we need to handle the CTM output correctly
        # final_predictions shape: (B, out_dims) - logits for each direction
        # target shape: (B, sequence_length) - each element is 0-3 for direction
        
        # Use the prediction directly since it's already the final output
        last_prediction = final_predictions  # Shape: (B, 4)
        
        # For maze task, we need to predict the next direction in the sequence
        # Since we only have one prediction per sample, we'll use it to predict the first direction
        # This is a simplified approach - in practice you might want to use all iterations
        target_first = target[:, 0]  # Take the first direction from each sequence
        
        # Compute loss
        loss = criterion(last_prediction, target_first)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        
        # Accuracy computation for direction prediction
        pred = last_prediction.argmax(dim=1)  # (B,)
        correct_predictions += (pred == target_first).sum().item()
        total_predictions += target.size(0)
        
        if batch_idx % 100 == 0:
            logging.info(f'{model_type} - Batch {batch_idx}/{len(dataloader)}, '
                        f'Loss: {loss.item():.4f}, '
                        f'Accuracy: {100. * correct_predictions / total_predictions:.2f}%')
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct_predictions / total_predictions
    
    return avg_loss, accuracy


def evaluate_model(model, dataloader, criterion, device, model_type):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    all_predictions = []
    all_targets = []
    all_certainties = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            predictions, certainties, _ = model(data)
            
            # Use the last prediction for evaluation
            final_predictions = predictions[:, :, -1]
            
            # For maze task, handle the CTM output correctly
            last_prediction = final_predictions  # Shape: (B, 4)
            target_first = target[:, 0]  # Take the first direction from each sequence
            
            # Compute loss
            loss = criterion(last_prediction, target_first)
            total_loss += loss.item()
            
            # Accuracy computation for direction prediction
            pred = last_prediction.argmax(dim=1)  # (B,)
            correct_predictions += (pred == target_first).sum().item()
            total_predictions += target.size(0)
            
            # Store for analysis
            all_predictions.append(final_predictions.cpu())
            all_targets.append(target.cpu())
            all_certainties.append(certainties[:, :, -1].cpu())  # Last certainty
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct_predictions / total_predictions
    
    return avg_loss, accuracy, all_predictions, all_targets, all_certainties


def plot_training_curves(baseline_losses, normalized_losses, baseline_accs, normalized_accs, save_dir):
    """Plot training curves for comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(baseline_losses, label='Baseline CTM', color='blue')
    ax1.plot(normalized_losses, label='Normalized CTM', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(baseline_accs, label='Baseline CTM', color='blue')
    ax2.plot(normalized_accs, label='Normalized CTM', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_certainty_comparison(baseline_certainties, normalized_certainties, save_dir):
    """Plot certainty distributions for comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Flatten certainty arrays
    baseline_certainty_flat = torch.cat(baseline_certainties, dim=0).flatten().numpy()
    normalized_certainty_flat = torch.cat(normalized_certainties, dim=0).flatten().numpy()
    
    # Histograms
    ax1.hist(baseline_certainty_flat, bins=50, alpha=0.7, label='Baseline CTM', color='blue')
    ax1.hist(normalized_certainty_flat, bins=50, alpha=0.7, label='Normalized CTM', color='red')
    ax1.set_xlabel('Certainty')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Certainty Distribution Comparison')
    ax1.legend()
    
    # Box plots
    ax2.boxplot([baseline_certainty_flat, normalized_certainty_flat], 
                labels=['Baseline CTM', 'Normalized CTM'])
    ax2.set_ylabel('Certainty')
    ax2.set_title('Certainty Statistics Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'certainty_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_synchronization_matrix(model, dataloader, device, model_type, save_dir):
    """Analyze synchronization matrix interpretability."""
    model.eval()
    
    # Collect synchronization matrices
    synch_matrices = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            _, _, synch_out = model(data)
            # Ensure all matrices have the same size by taking the first batch size
            if len(synch_matrices) == 0:
                # First batch - use this as reference size
                reference_size = synch_out.shape[0]
                synch_matrices.append(synch_out.cpu())
            else:
                # For subsequent batches, ensure they match the reference size
                if synch_out.shape[0] == reference_size:
                    synch_matrices.append(synch_out.cpu())
                else:
                    # Skip batches that don't match the reference size
                    # This typically happens with the last batch if it's smaller
                    continue
    
    # Average synchronization matrix
    if synch_matrices:
        avg_synch = torch.stack(synch_matrices).mean(dim=0)
    else:
        # Fallback if no valid matrices found
        logging.warning(f"No valid synchronization matrices found for {model_type}")
        return None
    
    # Plot as correlation matrix
    if avg_synch is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_synch.numpy(), cmap='RdBu_r', center=0, 
                    square=True, cbar_kws={'label': 'Synchronization'})
        plt.title(f'{model_type} CTM - Average Synchronization Matrix')
        plt.xlabel('Neuron Index')
        plt.ylabel('Neuron Index')
        plt.savefig(os.path.join(save_dir, f'{model_type.lower()}_synch_matrix.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    else:
        logging.warning(f"Skipping synchronization matrix plot for {model_type} due to no valid data")
    
    return avg_synch


def main():
    parser = argparse.ArgumentParser(description='Compare baseline and normalized CTM on maze task')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    # Model configuration
    config = {
        'iterations': 10,
        'd_model': 256,
        'd_input': 128,
        'heads': 8,
        'n_synch_out': 64,
        'n_synch_action': 64,
        'synapse_depth': 3,
        'memory_length': 8,
        'deep_nlms': True,
        'memory_hidden_dims': 64,
        'do_layernorm_nlm': False,
        'backbone_type': 'resnet18-2',
        'positional_embedding_type': 'learnable-fourier',
        'out_dims': 4,  # 4-direction maze navigation (up, down, left, right)
        'dropout': 0.1,
        'neuron_select_type': 'random-pairing',
        'normalization_decay': 0.01,
        'normalization_epsilon': 1e-5,
    }
    
    # Create datasets using MazeImageFolder
    which_maze = 'medium'  # Default to medium size mazes
    data_root = f'{args.data_dir}/{which_maze}'
    
    train_dataset = MazeImageFolder(
        root=f'{data_root}/train/', 
        which_set='train', 
        maze_route_length=100,  # Default route length
        expand_range=True
    )
    val_dataset = MazeImageFolder(
        root=f'{data_root}/test/', 
        which_set='test', 
        maze_route_length=100,  # Default route length
        expand_range=True
    )
    
    # Create separate dataloaders for each model to avoid graph conflicts
    train_loader_baseline = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader_normalized = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader_baseline = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader_normalized = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create models
    baseline_model = create_model('baseline', config, device)
    normalized_model = create_model('normalized', config, device)
    
    # Loss and optimizers
    criterion = nn.CrossEntropyLoss()
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=args.lr)
    normalized_optimizer = optim.Adam(normalized_model.parameters(), lr=args.lr)
    
    # Training tracking
    baseline_losses, normalized_losses = [], []
    baseline_accs, normalized_accs = [], []
    
    logger.info("Starting training comparison...")
    
    # Training loop - train models separately to avoid graph conflicts
    logger.info("Training baseline model...")
    for epoch in range(args.epochs):
        logger.info(f"Baseline Epoch {epoch+1}/{args.epochs}")
        
        # Train baseline
        baseline_loss, baseline_acc = train_epoch(
            baseline_model, train_loader_baseline, criterion, baseline_optimizer, device, 'Baseline'
        )
        baseline_losses.append(baseline_loss)
        baseline_accs.append(baseline_acc)
        
        # Validation
        baseline_val_loss, baseline_val_acc, _, _, baseline_certainties = evaluate_model(
            baseline_model, val_loader_baseline, criterion, device, 'Baseline'
        )
        
        logger.info(f"Baseline Epoch {epoch+1} - "
                   f"Loss={baseline_loss:.4f}, Acc={baseline_acc:.2f}%, "
                   f"Val Loss={baseline_val_loss:.4f}, Val Acc={baseline_val_acc:.2f}%")
    
    logger.info("Training normalized model...")
    for epoch in range(args.epochs):
        logger.info(f"Normalized Epoch {epoch+1}/{args.epochs}")
        
        # Train normalized
        normalized_loss, normalized_acc = train_epoch(
            normalized_model, train_loader_normalized, criterion, normalized_optimizer, device, 'Normalized'
        )
        normalized_losses.append(normalized_loss)
        normalized_accs.append(normalized_acc)
        
        # Validation
        normalized_val_loss, normalized_val_acc, _, _, normalized_certainties = evaluate_model(
            normalized_model, val_loader_normalized, criterion, device, 'Normalized'
        )
        
        logger.info(f"Normalized Epoch {epoch+1} - "
                   f"Loss={normalized_loss:.4f}, Acc={normalized_acc:.2f}%, "
                   f"Val Loss={normalized_val_loss:.4f}, Val Acc={normalized_val_acc:.2f}%")
    
    # Analysis and visualization
    logger.info("Generating analysis plots...")
    
    # Plot training curves
    plot_training_curves(baseline_losses, normalized_losses, 
                        baseline_accs, normalized_accs, args.output_dir)
    
    # Plot certainty comparison
    plot_certainty_comparison(baseline_certainties, normalized_certainties, args.output_dir)
    
    # Analyze synchronization matrices
    baseline_synch = analyze_synchronization_matrix(
        baseline_model, val_loader_baseline, device, 'Baseline', args.output_dir
    )
    normalized_synch = analyze_synchronization_matrix(
        normalized_model, val_loader_normalized, device, 'Normalized', args.output_dir
    )
    
    # Save results
    results = {
        'baseline_final_accuracy': baseline_accs[-1],
        'normalized_final_accuracy': normalized_accs[-1],
        'baseline_final_loss': baseline_losses[-1],
        'normalized_final_loss': normalized_losses[-1],
        'improvement_accuracy': normalized_accs[-1] - baseline_accs[-1],
        'improvement_loss': baseline_losses[-1] - normalized_losses[-1],
        'config': config,
        'training_curves': {
            'baseline_losses': baseline_losses,
            'normalized_losses': normalized_losses,
            'baseline_accuracies': baseline_accs,
            'normalized_accuracies': normalized_accs,
        }
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Training comparison completed!")
    logger.info(f"Final Baseline Accuracy: {baseline_accs[-1]:.2f}%")
    logger.info(f"Final Normalized Accuracy: {normalized_accs[-1]:.2f}%")
    logger.info(f"Accuracy Improvement: {normalized_accs[-1] - baseline_accs[-1]:.2f}%")


if __name__ == '__main__':
    main() 