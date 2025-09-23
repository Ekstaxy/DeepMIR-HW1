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
from experiments.logger_utils import create_experiment_loggers

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
            n_class=200  # Artist20 dataset
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
            min_lr=config.lr_scheduler.reduce_on_plateau.min_lr,
            verbose=config.lr_scheduler.reduce_on_plateau.verbose
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

    # TODO: Implement training loop for one epoch
    # 1. Iterate through train_loader with tqdm for progress bar
    # 2. Move data to device
    # 3. Forward pass
    # 4. Compute loss
    # 5. Backward pass and optimization
    # 6. Track metrics (loss, accuracy)
    # 7. Apply gradient clipping if configured
    # 8. Return epoch metrics (loss, accuracy)

    # Example structure:
    # with tqdm(train_loader, desc="Training") as pbar:
    #     for batch_idx, (data, target) in enumerate(pbar):
    #         data, target = data.to(device), target.to(device)
    #
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = criterion(output, target)
    #         loss.backward()
    #
    #         # Gradient clipping
    #         if config.training.get('gradient_clip_norm'):
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
    #
    #         optimizer.step()
    #
    #         # Update metrics
    #         total_loss += loss.item()
    #         _, predicted = output.max(1)
    #         total += target.size(0)
    #         correct += predicted.eq(target).sum().item()
    #
    #         # Update progress bar
    #         pbar.set_postfix({
    #             'Loss': loss.item(),
    #             'Acc': 100. * correct / total
    #         })

    raise NotImplementedError("TODO: Implement training epoch")


def validate_epoch(model, val_loader, criterion, device):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    # TODO: Implement validation loop
    # 1. Set model to eval mode (already done)
    # 2. Disable gradients with torch.no_grad()
    # 3. Iterate through val_loader
    # 4. Forward pass only
    # 5. Compute loss and metrics
    # 6. Collect predictions for top-k accuracy
    # 7. Return validation metrics

    # Example structure:
    # with torch.no_grad():
    #     with tqdm(val_loader, desc="Validation") as pbar:
    #         for data, target in pbar:
    #             data, target = data.to(device), target.to(device)
    #             output = model(data)
    #             loss = criterion(output, target)
    #
    #             total_loss += loss.item()
    #             _, predicted = output.max(1)
    #             total += target.size(0)
    #             correct += predicted.eq(target).sum().item()
    #
    #             # Store for top-k metrics
    #             all_predictions.append(output.cpu())
    #             all_targets.append(target.cpu())

    # Calculate top-1 and top-3 accuracy
    # Return metrics dictionary

    raise NotImplementedError("TODO: Implement validation epoch")


def calculate_top_k_accuracy(predictions, targets, k=3):
    """Calculate top-k accuracy."""
    # TODO: Implement top-k accuracy calculation
    # 1. Concatenate all predictions and targets
    # 2. Get top-k predictions for each sample
    # 3. Calculate accuracy
    # 4. Return top-1 and top-k accuracy

    raise NotImplementedError("TODO: Implement top-k accuracy calculation")


def train_model(model, train_loader, val_loader, config, device, tracker):
    """Main training loop."""
    logger.info("Starting training...")

    # Create optimizer, scheduler, and loss function
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    criterion = create_loss_function(config)

    # TODO: Implement main training loop
    # 1. Initialize best metrics for checkpointing
    # 2. Loop through epochs
    # 3. Train for one epoch
    # 4. Validate every config.training.validate_every epochs
    # 5. Update learning rate scheduler
    # 6. Save checkpoints
    # 7. Log metrics to tracker
    # 8. Implement early stopping
    # 9. Return trained model and best metrics

    # Example structure:
    # best_val_acc = 0.0
    # early_stopping_counter = 0
    #
    # for epoch in range(config.training.num_epochs):
    #     logger.info(f"\nEpoch {epoch+1}/{config.training.num_epochs}")
    #
    #     # Train
    #     train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, config)
    #
    #     # Validate
    #     if (epoch + 1) % config.training.validate_every == 0:
    #         val_metrics = validate_epoch(model, val_loader, criterion, device)
    #
    #         # Log metrics
    #         tracker.log_metrics({
    #             'epoch': epoch + 1,
    #             'train_loss': train_metrics['loss'],
    #             'train_acc': train_metrics['accuracy'],
    #             'val_loss': val_metrics['loss'],
    #             'val_acc': val_metrics['accuracy'],
    #             'val_top3_acc': val_metrics['top3_accuracy'],
    #             'learning_rate': optimizer.param_groups[0]['lr']
    #         }, step=epoch)
    #
    #         # Update scheduler
    #         if config.lr_scheduler.name == "reduce_on_plateau":
    #             scheduler.step(val_metrics['accuracy'])
    #         else:
    #             scheduler.step()
    #
    #         # Save checkpoint
    #         if val_metrics['accuracy'] > best_val_acc:
    #             best_val_acc = val_metrics['accuracy']
    #             tracker.save_checkpoint(model, optimizer, epoch, val_metrics)
    #             early_stopping_counter = 0
    #         else:
    #             early_stopping_counter += 1
    #
    #         # Early stopping
    #         if early_stopping_counter >= config.training.early_stopping_patience:
    #             logger.info("Early stopping triggered")
    #             break

    raise NotImplementedError("TODO: Implement main training loop")


def generate_test_predictions(model, test_loader, artist_to_id, config, device):
    """Generate predictions for test set."""
    logger.info("Generating test predictions...")

    # TODO: Implement test prediction generation
    # 1. Set model to eval mode
    # 2. Create id_to_artist mapping
    # 3. Iterate through test_loader
    # 4. Get model predictions
    # 5. Convert to top-3 artist names
    # 6. Format as required JSON structure
    # 7. Save predictions to file
    # 8. Return path to predictions file

    # Expected JSON format:
    # {
    #     "1": ["artist_name_1", "artist_name_2", "artist_name_3"],
    #     "2": ["artist_name_1", "artist_name_2", "artist_name_3"],
    #     ...
    # }

    raise NotImplementedError("TODO: Implement test prediction generation")


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
    main_logger, metrics_logger = create_experiment_loggers(config, experiment_name)
    main_logger.info("Starting deep learning training...")

    # Initialize experiment tracker
    tracker = ExperimentTracker(
        config=config,
        experiment_name=experiment_name,
        tags=["deep_learning", "cnn", "classification"],
        notes="Training CNN models for end-to-end singer classification",
        use_wandb=bool(config.wandb.project)
    )

    try:
        # TODO: Complete the main training pipeline
        # 1. Create data loaders
        # 2. Create model and move to device
        # 3. Log model information
        # 4. Train the model
        # 5. Generate test predictions
        # 6. Log final results

        # Example structure:
        # # Step 1: Create data loaders
        # train_loader, val_loader, artist_to_id = create_dataloaders(
        #     config, experiment_type="deep_learning"
        # )
        # test_loader, test_dataset = create_test_dataloader(
        #     config, experiment_type="deep_learning"
        # )
        #
        # # Step 2: Create model
        # model = create_model(config)
        # model = model.to(device)
        #
        # # Step 3: Log model info
        # tracker.log_model_info(model)
        #
        # # Step 4: Train
        # trained_model, best_metrics = train_model(
        #     model, train_loader, val_loader, config, device, tracker
        # )
        #
        # # Step 5: Generate predictions
        # predictions_path = generate_test_predictions(
        #     trained_model, test_loader, artist_to_id, config, device
        # )
        #
        # # Step 6: Log results
        # logger.info(f"Training completed!")
        # logger.info(f"Best validation accuracy: {best_metrics['accuracy']:.4f}")
        # logger.info(f"Test predictions saved to: {predictions_path}")

        raise NotImplementedError("TODO: Implement main training pipeline")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        # Finish experiment tracking
        tracker.finish()


if __name__ == "__main__":
    main()