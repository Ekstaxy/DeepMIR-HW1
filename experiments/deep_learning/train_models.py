#!/usr/bin/env python3
"""
Main training script for deep learning models.
Train CNN models for end-to-end singer classification.
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import deep learning models
from models.deep_learning.dl_models import ShortChunkCNN, ShortChunkCNN_Res

# Import data utilities
from data.dataloaders import create_dataloaders, create_test_dataloader
from data.utils import get_artist_mapping

# Import experiment utilities
from experiments.tracking import ExperimentTracker

logger = logging.getLogger(__name__)


def create_model(config):
    """Create and initialize the deep learning model."""
    logger.info("Creating model...")

    if config.model.name == "ShortChunkCNN":
        model = ShortChunkCNN(
            n_channels=config.model.n_channels,
            sample_rate=config.audio.sample_rate,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            f_min=config.audio.f_min,
            f_max=config.audio.f_max,
            n_mels=config.audio.n_mels,
            n_class=20  # Artist20 dataset
        )
    elif config.model.name == "ShortChunkCNN_Res":
        model = ShortChunkCNN_Res(
            n_channels=config.model.n_channels,
            sample_rate=config.audio.sample_rate,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            f_min=config.audio.f_min,
            f_max=config.audio.f_max,
            n_mels=config.audio.n_mels,
            n_class=20  # Artist20 dataset
        )
    else:
        raise ValueError(f"Unknown model: {config.model.name}")

    logger.info(f"Created {config.model.name} with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def create_optimizer(model, config):
    """Create optimizer for training."""
    logger.info("Creating optimizer...")

    optimizer_name = config.optimizer.name.lower()

    if optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps
        )
    elif optimizer_name == "sgd":
        # Use parameters from alternatives section or defaults
        momentum = config.optimizer.alternatives.sgd.get('momentum', 0.9)
        nesterov = config.optimizer.alternatives.sgd.get('nesterov', True)

        optimizer = optim.SGD(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            momentum=momentum,
            nesterov=nesterov
        )
    elif optimizer_name == "adamw":
        # Use parameters from alternatives section or defaults
        betas = config.optimizer.alternatives.adamw.get('betas', [0.9, 0.999])
        eps = config.optimizer.alternatives.adamw.get('eps', 1e-8)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            betas=betas,
            eps=eps
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer.name}")

    logger.info(f"Created {config.optimizer.name} optimizer with lr={config.optimizer.learning_rate}")
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler."""
    logger.info("Creating learning rate scheduler...")

    # TODO: Implement scheduler creation based on config
    # Support different schedulers: ReduceLROnPlateau, CosineAnnealingLR, StepLR
    # Use parameters from config.lr_scheduler
    if config.lr_scheduler.name == "none":
        scheduler = None
    elif config.lr_scheduler.name == "constant":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    elif config.lr_scheduler.name == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.lr_scheduler.reduce_on_plateau.mode,
            factor=config.lr_scheduler.reduce_on_plateau.factor,
            patience=config.lr_scheduler.reduce_on_plateau.patience,
            min_lr=config.lr_scheduler.reduce_on_plateau.min_lr
        )
    elif config.lr_scheduler.name == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.lr_scheduler.alternatives.cosine_annealing.T_max,
            eta_min=config.lr_scheduler.alternatives.cosine_annealing.eta_min,
            last_epoch=-1
        )
    elif config.lr_scheduler.name == "step_lr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_scheduler.alternatives.step_lr.step_size,
            gamma=config.lr_scheduler.alternatives.step_lr.gamma,
            last_epoch=-1
        )
    else:
        logger.warning(f"Unknown scheduler: {config.lr_scheduler.name}, using no scheduler.")
        scheduler = None
    return scheduler


def create_loss_function(config):
    """Create loss function for training."""
    logger.info("Creating loss function...")

    if config.loss.name == "none":
        loss_func = None
    elif config.loss.name == "cross_entropy":
        loss_func = nn.CrossEntropyLoss()
    else:
        logger.warning(f"Unknown loss function: {config.loss.name}, using CrossEntropyLoss.")
        loss_func = nn.CrossEntropyLoss()

    return loss_func


def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    with tqdm(train_loader, desc="Training") as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping
            if config.training.get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)

            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': loss.item(),
                'Acc': 100. * correct / total
            })

    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def validate_epoch(model, val_loader, criterion, device):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        with tqdm(val_loader, desc="Validation") as pbar:
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Store for top-k metrics
                all_predictions.append(output.cpu())
                all_targets.append(target.cpu())

                # Update progress bar
                pbar.set_postfix({
                    'Loss': loss.item(),
                    'Acc': 100. * correct / total
                })

    # Calculate top-1 and top-3 accuracy
    top1_acc, top3_acc = calculate_top_k_accuracy(all_predictions, all_targets, k=3)

    avg_loss = total_loss / len(val_loader)

    return {
        'loss': avg_loss,
        'accuracy': top1_acc,
        'top3_accuracy': top3_acc
    }


def calculate_top_k_accuracy(predictions, targets, k=3):
    """Calculate top-k accuracy."""
    # Concatenate all predictions and targets
    all_predictions = torch.cat(predictions, dim=0)
    all_targets = torch.cat(targets, dim=0)

    # Get top-k predictions for each sample
    _, top_k_pred = all_predictions.topk(k, dim=1, largest=True, sorted=True)

    # Calculate top-1 accuracy
    top1_correct = top_k_pred[:, 0].eq(all_targets).sum().item()
    top1_accuracy = 100.0 * top1_correct / all_targets.size(0)

    # Calculate top-k accuracy
    correct = 0
    for i in range(all_targets.size(0)):
        if all_targets[i] in top_k_pred[i]:
            correct += 1
    topk_accuracy = 100.0 * correct / all_targets.size(0)

    return top1_accuracy, topk_accuracy


def train_model(model, train_loader, val_loader, config, device, tracker):
    """Main training loop."""
    logger.info("Starting training...")

    # Create optimizer, scheduler, and loss function
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    criterion = create_loss_function(config)

    # Training will use standard logger for progress

    # Initialize best metrics for checkpointing
    best_val_acc = 0.0
    early_stopping_counter = 0
    best_metrics = {}

    for epoch in range(config.training.num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, config)

        # Validate
        if (epoch + 1) % config.training.validate_every == 0:
            val_metrics = validate_epoch(model, val_loader, criterion, device)

            # Log epoch summary
            logger.info(f"Epoch {epoch + 1:3d} | train_loss={train_metrics['loss']:.4f} | train_acc={train_metrics['accuracy']:.4f} | val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | val_top3_acc={val_metrics['top3_accuracy']:.4f}")

            # Log metrics to tracker
            tracker.log_metrics({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_top3_acc': val_metrics['top3_accuracy'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)

            # Update scheduler
            if scheduler is not None:
                if config.lr_scheduler.name == "reduce_on_plateau":
                    scheduler.step(val_metrics['accuracy'])
                else:
                    scheduler.step()

            # Save checkpoint if best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_metrics = val_metrics.copy()
                best_metrics['epoch'] = epoch + 1
                tracker.save_checkpoint(model, optimizer, epoch, val_metrics)
                early_stopping_counter = 0
                logger.info(f"New best model: {best_val_acc:.4f}% accuracy")
            else:
                early_stopping_counter += 1

            # Early stopping
            if early_stopping_counter >= config.training.early_stopping_patience:
                logger.info("Early stopping triggered")
                break

        # Log training metrics only (no validation)
        else:
            logger.info(f"Epoch {epoch + 1:3d} | train_loss={train_metrics['loss']:.4f} | train_acc={train_metrics['accuracy']:.4f}")

            tracker.log_metrics({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)

    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}%")
    return model, best_metrics


def generate_test_predictions(model, test_loader, artist_to_id, config, device):
    """Generate predictions for test set."""
    logger.info("Generating test predictions...")

    # Set model to eval mode
    model.eval()

    # Create id_to_artist mapping
    id_to_artist = {v: k for k, v in artist_to_id.items()}

    # Store predictions
    predictions = {}

    with torch.no_grad():
        with tqdm(test_loader, desc="Test Predictions") as pbar:
            for batch_data, test_ids in pbar:
                batch_data = batch_data.to(device)

                # Get model predictions
                outputs = model(batch_data)

                # Get top-3 predictions for each sample in batch
                _, top3_indices = outputs.topk(3, dim=1, largest=True, sorted=True)

                # Process each sample in the batch
                for i, test_id in enumerate(test_ids):
                    # Convert indices to artist names
                    top3_artists = []
                    for idx in top3_indices[i]:
                        artist_name = id_to_artist.get(idx.item(), f"Unknown_{idx.item()}")
                        top3_artists.append(artist_name)

                    # Store prediction (test_id should be like "001", "002", etc.)
                    predictions[test_id] = top3_artists

    # Create predictions directory
    predictions_dir = Path(config.paths.predictions)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions to JSON file
    predictions_path = predictions_dir / "deep_learning_test_predictions.json"

    import json
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    logger.info(f"Generated predictions for {len(predictions)} test samples")
    logger.info(f"Test predictions saved to: {predictions_path}")

    return predictions_path


@hydra.main(version_base=None, config_path="../../configs", config_name="deep_learning/baseline_config")
def main(cfg: DictConfig):
    """Main training function."""
    # Extract config
    if 'deep_learning' in cfg:
        config = cfg.deep_learning
    else:
        config = cfg

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Setup logging and tracking
    experiment_name = f"deep_learning_{config.experiment.name}"
    logger.info("Starting deep learning training...")

    # Initialize experiment tracker
    tracker = ExperimentTracker(
        config=config,
        experiment_name=experiment_name,
        tags=["deep_learning", "cnn", "classification"],
        notes="Training CNN models for end-to-end singer classification",
        use_wandb=bool(config.wandb.project)
    )

    try:
        # Step 1: Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, artist_to_id = create_dataloaders(
            config, experiment_type="deep_learning"
        )
        test_loader, test_dataset = create_test_dataloader(
            config, experiment_type="deep_learning"
        )

        logger.info(f"Train loader: {len(train_loader)} batches")
        logger.info(f"Val loader: {len(val_loader)} batches")
        logger.info(f"Test loader: {len(test_loader)} batches")
        logger.info(f"Found {len(artist_to_id)} artists")

        # Step 2: Create model
        logger.info("Creating model...")
        model = create_model(config)
        model = model.to(device)

        # Step 3: Log model info
        tracker.log_model_info(model)

        # Log training configuration
        logger.info("Training configuration:")
        logger.info(f"  Epochs: {config.training.num_epochs}")
        logger.info(f"  Batch size: {config.training.batch_size}")
        logger.info(f"  Learning rate: {config.optimizer.learning_rate}")
        logger.info(f"  Optimizer: {config.optimizer.name}")
        logger.info(f"  Scheduler: {config.lr_scheduler.name}")
        logger.info(f"  Early stopping patience: {config.training.early_stopping_patience}")

        # Step 4: Train
        logger.info("Starting training...")
        trained_model, best_metrics = train_model(
            model, train_loader, val_loader, config, device, tracker
        )

        # Step 5: Generate predictions
        logger.info("Generating test predictions...")
        predictions_path = generate_test_predictions(
            trained_model, test_loader, artist_to_id, config, device
        )

        # Step 6: Log final results
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Best validation accuracy: {best_metrics['accuracy']:.4f}%")
        logger.info(f"Best validation top-3 accuracy: {best_metrics['top3_accuracy']:.4f}%")
        logger.info(f"Test predictions saved to: {predictions_path}")
        logger.info("Model checkpoints saved in experiment tracker")

        # Calculate final score (same as evaluation metric)
        final_score = best_metrics['accuracy'] + 0.5 * best_metrics['top3_accuracy']
        logger.info(f"Final score: {final_score:.4f}")

        # Log final score to tracker
        tracker.log_metrics({
            'final_score': final_score,
            'best_epoch': best_metrics['epoch'],
            'best_val_accuracy': best_metrics['accuracy'],
            'best_val_top3_accuracy': best_metrics['top3_accuracy']
        })

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        # Finish experiment tracking
        tracker.finish()


if __name__ == "__main__":
    main()