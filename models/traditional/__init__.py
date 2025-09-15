"""
Traditional machine learning models for singer classification.
"""

from .feature_extractors import (
    MFCCExtractor,
    SpectralFeaturesExtractor,
    ChromaExtractor,
    TonnetzExtractor,
    TempoExtractor,
    AudioFeatureExtractor,
    extract_features_from_dataset
)

__all__ = [
    'MFCCExtractor',
    'SpectralFeaturesExtractor',
    'ChromaExtractor',
    'TonnetzExtractor',
    'TempoExtractor',
    'AudioFeatureExtractor',
    'extract_features_from_dataset'
]