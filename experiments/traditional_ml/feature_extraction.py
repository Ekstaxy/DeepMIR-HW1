"""
Feature extraction script for traditional ML experiments.
Template for extracting and saving audio features.
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import load_experiment_config, setup_device
from data import create_dataloaders
from models.traditional import AudioFeatureExtractor, extract_features_from_dataset
from experiments import ExperimentTracker

logger = logging.getLogger(__name__)


def extract_features_main(
    config_name: str = "baseline",
    force_recompute: bool = False,
    save_features: bool = True
):
    """
    Main function for feature extraction.

    Args:
        config_name: Name of the configuration to use
        force_recompute: Force recomputation of features
        save_features: Save extracted features to disk
    """
    # TODO: Load configuration
    # config = load_experiment_config("traditional_ml", config_name)

    # TODO: Setup device (CPU for traditional ML)
    # device = setup_device(config)

    # TODO: Create experiment tracker
    # with ExperimentTracker(
    #     config=config,
    #     experiment_name=f"feature_extraction_{config_name}",
    #     tags=["feature_extraction", "traditional_ml"],
    #     use_wandb=False
    # ) as tracker:

        # TODO: Create data loaders
        # train_loader, val_loader, artist_to_id = create_dataloaders(
        #     config,
        #     experiment_type="traditional_ml"
        # )

        # TODO: Initialize feature extractor
        # feature_extractor = AudioFeatureExtractor(config)

        # TODO: Extract features from training set
        # train_features, train_labels = extract_features_from_dataset(
        #     train_loader.dataset,
        #     config,
        #     save_path="data/processed/train_features.npz" if save_features else None
        # )

        # TODO: Extract features from validation set
        # val_features, val_labels = extract_features_from_dataset(
        #     val_loader.dataset,
        #     config,
        #     save_path="data/processed/val_features.npz" if save_features else None
        # )

        # TODO: Log feature statistics
        # tracker.log_metrics({
        #     "train_samples": len(train_features),
        #     "val_samples": len(val_features),
        #     "feature_dim": train_features.shape[1],
        #     "num_classes": len(artist_to_id)
        # })

        # TODO: Log feature distributions
        # tracker.log_metrics({
        #     "train_feature_mean": np.mean(train_features),
        #     "train_feature_std": np.std(train_features),
        #     "val_feature_mean": np.mean(val_features),
        #     "val_feature_std": np.std(val_features)
        # })

    pass


def analyze_features(features_path: str):
    """
    Analyze extracted features.

    Args:
        features_path: Path to saved features
    """
    # TODO: Load features
    # features, labels, metadata = load_features(features_path)

    # TODO: Compute feature statistics
    # TODO: Create visualizations
    # TODO: Analyze class separability
    pass


def visualize_features(features_path: str, output_dir: str):
    """
    Create visualizations of extracted features.

    Args:
        features_path: Path to saved features
        output_dir: Directory to save visualizations
    """
    # TODO: Load features
    # TODO: Create t-SNE visualization
    # TODO: Create PCA visualization
    # TODO: Create correlation heatmap
    # TODO: Create feature importance plots
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract audio features for traditional ML")
    parser.add_argument(
        "--config",
        default="baseline",
        help="Configuration name to use"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of features"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save features to disk"
    )
    parser.add_argument(
        "--analyze",
        type=str,
        help="Path to features file to analyze"
    )
    parser.add_argument(
        "--visualize",
        type=str,
        help="Path to features file to visualize"
    )

    args = parser.parse_args()

    if args.analyze:
        analyze_features(args.analyze)
    elif args.visualize:
        visualize_features(args.visualize, "results/visualizations/features")
    else:
        extract_features_main(
            config_name=args.config,
            force_recompute=args.force_recompute,
            save_features=not args.no_save
        )