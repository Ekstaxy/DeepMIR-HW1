#!/usr/bin/env python3
"""
Generate test predictions for traditional ML models.
Create JSON predictions for final submission.
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import hydra
from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.traditional.ml_models import create_model, generate_test_predictions
from models.traditional.feature_extractors import AudioFeatureExtractor
from experiments.logger_utils import setup_logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../configs/traditional_ml", config_name="baseline_config")
def main(config: DictConfig):
    """Generate test predictions from saved models."""

    setup_logging(config)
    logger.info("Generating test predictions...")

    try:
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

        # Extract features from test files
        test_dir = Path(config.dataset.test_dir)
        test_files = sorted(list(test_dir.glob("*.mp3")))

        if not test_files:
            logger.error(f"No test files found in {test_dir}")
            return

        logger.info(f"Found {len(test_files)} test files")

        feature_extractor = AudioFeatureExtractor(config)
        X_test = []
        test_filenames = []

        for test_file in test_files:
            logger.info(f"Processing: {test_file.name}")
            features = feature_extractor.extract_features(str(test_file))
            X_test.append(features)
            test_filenames.append(test_file.stem)

        X_test = np.array(X_test)
        logger.info(f"Test features shape: {X_test.shape}")

        # Generate predictions for each model
        predictions_dir = Path(config.paths.predictions)
        predictions_dir.mkdir(parents=True, exist_ok=True)

        for model_type, model in models.items():
            predictions_path = predictions_dir / f"{model_type}_test_predictions.json"
            logger.info(f"Generating predictions for {model_type}...")
            generate_test_predictions(model, X_test, test_filenames, predictions_path)
            logger.info(f"Predictions saved to: {predictions_path}")

        logger.info("Test prediction generation complete!")

    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        raise

if __name__ == "__main__":
    main()