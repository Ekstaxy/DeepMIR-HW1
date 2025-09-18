#!/usr/bin/env python3
"""
Model evaluation script for traditional ML models.
Evaluate trained models and generate comprehensive metrics.
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.traditional.ml_models import create_model, ModelEvaluator
from models.traditional.feature_extractors import AudioFeatureExtractor, extract_features_from_dataset
from data.datasets import Artist20Dataset
from experiments.tracking import ExperimentTracker
from experiments.logger_utils import create_experiment_loggers

logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, labels, model_type, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_type.upper()}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix saved to: {save_path}")

@hydra.main(version_base=None, config_path="../../configs/traditional_ml", config_name="baseline_config")
def main(config: DictConfig):
    """Evaluate all trained ML models."""

    experiment_name = f"model_evaluation_{config.experiment.name}"
    main_logger, metrics_logger = create_experiment_loggers(config, experiment_name)
    main_logger.info("Starting model evaluation...")

    # Initialize experiment tracker
    tracker = ExperimentTracker(
        config=config,
        experiment_name=f"model_evaluation_{config.experiment.name}",
        tags=["traditional_ml", "evaluation"],
        notes="Comprehensive evaluation of trained traditional ML models",
        use_wandb=bool(config.wandb.project)
    )

    try:
        # Load validation dataset
        logger.info("Loading validation dataset...")
        val_dataset = Artist20Dataset(config.dataset.val_json, config)
        logger.info(f"Loaded {len(val_dataset)} validation samples")

        # Extract features
        logger.info("Extracting features...")
        X_val, y_val = extract_features_from_dataset(val_dataset, config)
        logger.info(f"Validation features shape: {X_val.shape}")

        # Load saved models
        checkpoints_dir = Path(config.paths.checkpoints) / "traditional_ml"
        model_types = ['knn', 'svm', 'random_forest']

        models = {}
        for model_type in model_types:
            model_path = checkpoints_dir / f"{model_type}_model.pkl"

            if model_path.exists():
                logger.info(f"Loading {model_type} model from {model_path}")
                model = create_model(model_type, config)
                model.load_model(model_path)
                models[model_type] = model
            else:
                logger.warning(f"Model file not found: {model_path}")

        if not models:
            logger.error("No trained models found!")
            return

        # Initialize evaluator
        evaluator = ModelEvaluator(config)

        # Evaluation results
        all_results = {}
        all_metrics = {}

        # Create results directory
        results_dir = Path(config.paths.results) / "model_evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)

        for model_type, model in models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating {model_type.upper()} Model")
            logger.info(f"{'='*50}")

            try:
                # Evaluate model
                metrics = evaluator.evaluate_model(model, X_val, y_val)
                all_metrics[model_type] = metrics

                # Get predictions for confusion matrix
                y_pred = model.predict(X_val)

                # Get unique labels
                unique_labels = sorted(list(set(y_val)))

                # Generate classification report
                class_report = classification_report(y_val, y_pred, labels=unique_labels)
                logger.info(f"Classification Report for {model_type}:")
                logger.info(f"\n{class_report}")

                # Plot confusion matrix
                cm_path = results_dir / f"{model_type}_confusion_matrix.png"
                plot_confusion_matrix(y_val, y_pred, unique_labels, model_type, cm_path)

                # Log metrics
                logger.info(f"Results for {model_type}:")
                logger.info(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
                logger.info(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
                logger.info(f"  Final Score: {metrics['final_score']:.4f}")

                # Log to tracker
                tracker.log_metrics({
                    f'{model_type}/eval_top1_accuracy': metrics['top1_accuracy'],
                    f'{model_type}/eval_top3_accuracy': metrics['top3_accuracy'],
                    f'{model_type}/eval_final_score': metrics['final_score']
                })

                # Log confusion matrix image
                tracker.log_image(str(cm_path), f'{model_type}/confusion_matrix', f'Confusion Matrix - {model_type.upper()}')

                all_results[model_type] = {
                    'metrics': metrics,
                    'classification_report': class_report
                }

            except Exception as e:
                logger.error(f"Failed to evaluate {model_type}: {e}")
                continue

        # Print final comparison
        logger.info(f"\n{'='*50}")
        logger.info("MODEL EVALUATION COMPLETE")
        logger.info(f"{'='*50}")

        logger.info("Final Model Comparison:")
        for model_type, metrics in all_metrics.items():
            logger.info(f"{model_type.upper()}:")
            logger.info(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
            logger.info(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
            logger.info(f"  Final Score: {metrics['final_score']:.4f}")

        # Find best model
        best_model_type = max(all_metrics.keys(), key=lambda k: all_metrics[k]['final_score'])
        best_score = all_metrics[best_model_type]['final_score']
        logger.info(f"\nBest Model: {best_model_type.upper()} (Score: {best_score:.4f})")

        # Save evaluation results
        import json
        results_path = results_dir / "evaluation_results.json"

        # Convert results to JSON-serializable format
        json_results = {}
        for model_type, results in all_results.items():
            json_results[model_type] = {
                'metrics': {k: float(v) for k, v in results['metrics'].items()},
                'classification_report': results['classification_report']
            }

        json_results['best_model'] = {
            'model_type': best_model_type,
            'score': float(best_score)
        }

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Evaluation results saved to: {results_path}")

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise

    finally:
        # Finish experiment tracking
        tracker.finish()

if __name__ == "__main__":
    main()