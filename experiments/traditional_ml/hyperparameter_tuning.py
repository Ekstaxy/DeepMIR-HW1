#!/usr/bin/env python3
"""
Hyperparameter tuning script for traditional ML models.
Optimize model parameters using GridSearchCV.
"""

import os
import sys
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.traditional.ml_models import HyperparameterTuner
from models.traditional.feature_extractors import AudioFeatureExtractor, extract_features_from_dataset
from data.datasets import Artist20Dataset
from experiments.tracking import ExperimentTracker
from experiments.logger_utils import setup_logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../configs/traditional_ml", config_name="baseline_config")
def main(config: DictConfig):
    """Run hyperparameter tuning for all ML models."""

    setup_logging(config)
    logger.info("Starting hyperparameter tuning...")

    # Initialize experiment tracker
    tracker = ExperimentTracker(
        config=config,
        experiment_name=f"hyperparameter_tuning_{config.experiment.name}",
        tags=["traditional_ml", "hyperparameter_tuning"],
        notes="Hyperparameter tuning for traditional ML models",
        use_wandb=bool(config.wandb.project)
    )

    try:
        # Load training dataset
        logger.info("Loading training dataset...")
        train_dataset = Artist20Dataset(config.dataset.train_json, config)
        logger.info(f"Loaded {len(train_dataset)} training samples")

        # Extract features
        logger.info("Extracting features...")
        X_train, y_train = extract_features_from_dataset(train_dataset, config)
        logger.info(f"Training features shape: {X_train.shape}")
        logger.info(f"Number of unique classes: {len(set(y_train))}")

        # Initialize hyperparameter tuner
        tuner = HyperparameterTuner(config)

        # Model types to tune
        model_types = ['knn', 'svm', 'random_forest']
        all_results = {}

        for model_type in model_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Tuning {model_type.upper()} Hyperparameters")
            logger.info(f"{'='*50}")

            try:
                if model_type == 'knn':
                    results = tuner.tune_knn(X_train, y_train)
                elif model_type == 'svm':
                    results = tuner.tune_svm(X_train, y_train)
                elif model_type == 'random_forest':
                    results = tuner.tune_random_forest(X_train, y_train)

                all_results[model_type] = results

                # Log results
                logger.info(f"Best parameters for {model_type}:")
                for param, value in results['best_params'].items():
                    logger.info(f"  {param}: {value}")
                logger.info(f"Best cross-validation score: {results['best_score']:.4f}")

                # Log to tracker
                tracker.log_metrics({
                    f'{model_type}/best_cv_score': results['best_score'],
                    f'{model_type}/best_params': str(results['best_params'])
                })

            except Exception as e:
                logger.error(f"Failed to tune {model_type}: {e}")
                continue

        # Print summary
        logger.info(f"\n{'='*50}")
        logger.info("HYPERPARAMETER TUNING COMPLETE")
        logger.info(f"{'='*50}")

        for model_type, results in all_results.items():
            logger.info(f"{model_type.upper()}:")
            logger.info(f"  Best CV Score: {results['best_score']:.4f}")
            logger.info(f"  Best Params: {results['best_params']}")

        # Save results
        results_dir = Path(config.paths.results) / "hyperparameter_tuning"
        results_dir.mkdir(parents=True, exist_ok=True)

        import json
        results_path = results_dir / "tuning_results.json"

        # Convert results to JSON-serializable format
        json_results = {}
        for model_type, results in all_results.items():
            json_results[model_type] = {
                'best_score': float(results['best_score']),
                'best_params': results['best_params']
            }

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Tuning results saved to: {results_path}")

    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        raise

    finally:
        # Finish experiment tracking
        tracker.finish()

if __name__ == "__main__":
    main()