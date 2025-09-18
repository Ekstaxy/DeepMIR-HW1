"""
Dataset classes for the Singer Classification project.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Callable
import logging
import numpy as np

from .utils import (
    load_audio_file,
    validate_audio_file,
    get_artist_mapping,
    extract_artist_from_path,
    get_test_files
)
from .transforms import ComposeTransforms

logger = logging.getLogger(__name__)


class Artist20Dataset(Dataset):
    """Dataset class for Artist20 train/validation data."""

    def __init__(
        self,
        json_file_path: Union[str, Path],
        root_dir: Union[str, Path],
        artist_to_id: Optional[Dict[str, int]] = None,
        transform: Optional[Callable] = None,
        sample_rate: int = 16000,
        max_duration: Optional[float] = None,
        cache_audio: bool = False,
        validate_files: bool = True
    ):
        """
        Initialize Artist20Dataset.

        Args:
            json_file_path: Path to JSON file containing file paths
            root_dir: Root directory for resolving relative paths
            artist_to_id: Dictionary mapping artist names to IDs
            transform: Optional transform to apply to audio
            sample_rate: Sample rate for audio loading
            max_duration: Maximum duration in seconds
            cache_audio: Cache loaded audio in memory
            validate_files: Validate audio files during initialization
        """
        self.json_file_path = Path(json_file_path)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.cache_audio = cache_audio
        self.audio_cache = {} if cache_audio else None

        # Load file paths from JSON
        with open(self.json_file_path, 'r') as f:
            self.file_paths = json.load(f)

        # Convert relative paths to absolute
        self.absolute_paths = []
        for file_path in self.file_paths:
            if file_path.startswith('./'):
                abs_path = self.root_dir / file_path[2:]  # Remove './'
            else:
                abs_path = self.root_dir / file_path
            self.absolute_paths.append(str(abs_path))

        # Create or use provided artist mapping
        if artist_to_id is None:
            # Create mapping from training data
            train_json = self.root_dir / "train.json"
            if train_json.exists():
                self.artist_to_id = get_artist_mapping(train_json)
            else:
                # Fallback: create mapping from current file paths
                self.artist_to_id = self._create_artist_mapping()
        else:
            self.artist_to_id = artist_to_id

        self.id_to_artist = {v: k for k, v in self.artist_to_id.items()}
        self.num_classes = len(self.artist_to_id)

        # Extract labels
        self.labels = []
        self.valid_indices = []

        for idx, file_path in enumerate(self.file_paths):
            try:
                artist = extract_artist_from_path(file_path)
                if artist in self.artist_to_id:
                    label = self.artist_to_id[artist]
                    self.labels.append(label)
                    self.valid_indices.append(idx)
                else:
                    logger.warning(f"Unknown artist '{artist}' in {file_path}")
            except Exception as e:
                logger.warning(f"Failed to extract artist from {file_path}: {e}")

        # Validate audio files if requested
        if validate_files:
            self._validate_audio_files()

        logger.info(f"Loaded {len(self.valid_indices)} valid samples from {self.json_file_path}")
        logger.info(f"Found {self.num_classes} artists: {list(self.artist_to_id.keys())}")

        # Print label distribution
        label_counts = np.bincount(self.labels, minlength=self.num_classes)
        logger.info(f"Label distribution: {dict(zip(self.id_to_artist.values(), label_counts))}")

    def _create_artist_mapping(self) -> Dict[str, int]:
        """Create artist mapping from current file paths."""
        artists = set()
        for file_path in self.file_paths:
            try:
                artist = extract_artist_from_path(file_path)
                artists.add(artist)
            except Exception as e:
                logger.warning(f"Failed to extract artist from {file_path}: {e}")

        sorted_artists = sorted(list(artists))
        return {artist: idx for idx, artist in enumerate(sorted_artists)}

    def _validate_audio_files(self):
        """Validate that all audio files can be loaded."""
        invalid_indices = []
        for i, idx in enumerate(self.valid_indices):
            file_path = self.absolute_paths[idx]
            if not validate_audio_file(file_path):
                logger.warning(f"Invalid audio file: {file_path}")
                invalid_indices.append(i)

        # Remove invalid files
        for i in reversed(invalid_indices):
            self.valid_indices.pop(i)
            self.labels.pop(i)

        if invalid_indices:
            logger.info(f"Removed {len(invalid_indices)} invalid audio files")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (audio_data, label)
        """
        if idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        file_idx = self.valid_indices[idx]
        file_path = self.absolute_paths[file_idx]
        label = self.labels[idx]

        # Load audio (with caching if enabled)
        if self.cache_audio and file_path in self.audio_cache:
            audio = self.audio_cache[file_path]
        else:
            try:
                audio, sr = load_audio_file(
                    file_path,
                    sample_rate=self.sample_rate,
                    duration=self.max_duration
                )
                if self.cache_audio:
                    self.audio_cache[file_path] = audio
            except Exception as e:
                logger.error(f"Failed to load audio {file_path}: {e}")
                # Return zeros as fallback
                target_length = int(self.sample_rate * (self.max_duration or 30.0))
                audio = np.zeros(target_length, dtype=np.float32)

        # Apply transforms
        if self.transform:
            audio = self.transform(audio)

        # Convert to tensor if needed
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio).float()

        return audio, label

    def get_artist_name(self, label: int) -> str:
        """Get artist name from label."""
        return self.id_to_artist.get(label, f"Unknown_{label}")

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced dataset."""
        label_counts = np.bincount(self.labels, minlength=self.num_classes)
        total_samples = len(self.labels)

        # Inverse frequency weighting
        weights = total_samples / (self.num_classes * label_counts + 1e-6)
        return torch.from_numpy(weights).float()

    def get_sample_info(self, idx: int) -> Dict:
        """Get detailed information about a sample."""
        if idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of range")

        file_idx = self.valid_indices[idx]
        file_path = self.absolute_paths[file_idx]
        label = self.labels[idx]
        artist = self.get_artist_name(label)

        return {
            'index': idx,
            'file_path': file_path,
            'label': label,
            'artist': artist,
            'relative_path': self.file_paths[file_idx]
        }

class Artist20TestDataset(Dataset):
    """Dataset class for Artist20 test data."""

    def __init__(
        self,
        test_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        sample_rate: int = 16000,
        max_duration: Optional[float] = None,
        cache_audio: bool = False
    ):
        """
        Initialize Artist20TestDataset.

        Args:
            test_dir: Directory containing test audio files
            transform: Optional transform to apply to audio
            sample_rate: Sample rate for audio loading
            max_duration: Maximum duration in seconds
            cache_audio: Cache loaded audio in memory
        """
        self.test_dir = Path(test_dir)
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.cache_audio = cache_audio
        self.audio_cache = {} if cache_audio else None

        # Get test files in order (001.mp3 to 233.mp3)
        self.file_paths = get_test_files(test_dir)

        # Extract test IDs from filenames
        self.test_ids = []
        for file_path in self.file_paths:
            filename = Path(file_path).stem  # e.g., "001" from "001.mp3"
            self.test_ids.append(filename)

        logger.info(f"Loaded {len(self.file_paths)} test samples from {test_dir}")

    def __len__(self) -> int:
        """Return number of test samples."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a test sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (audio_data, test_id)
        """
        if idx >= len(self.file_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        file_path = self.file_paths[idx]
        test_id = self.test_ids[idx]

        # Load audio (with caching if enabled)
        if self.cache_audio and file_path in self.audio_cache:
            audio = self.audio_cache[file_path]
        else:
            try:
                audio, sr = load_audio_file(
                    file_path,
                    sample_rate=self.sample_rate,
                    duration=self.max_duration
                )
                if self.cache_audio:
                    self.audio_cache[file_path] = audio
            except Exception as e:
                logger.error(f"Failed to load test audio {file_path}: {e}")
                # Return zeros as fallback
                target_length = int(self.sample_rate * (self.max_duration or 30.0))
                audio = np.zeros(target_length, dtype=np.float32)

        # Apply transforms
        if self.transform:
            audio = self.transform(audio)

        # Convert to tensor if needed
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio).float()

        return audio, test_id

    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a test sample."""
        if idx >= len(self.file_paths):
            raise IndexError(f"Index {idx} out of range")

        return {
            'index': idx,
            'file_path': self.file_paths[idx],
            'test_id': self.test_ids[idx]
        }

class CachedDataset(Dataset):
    """Dataset wrapper that caches preprocessed features."""

    def __init__(
        self,
        base_dataset: Dataset,
        cache_dir: Union[str, Path],
        cache_name: str,
        force_recompute: bool = False
    ):
        """
        Initialize cached dataset.

        Args:
            base_dataset: Base dataset to wrap
            cache_dir: Directory to store cache files
            cache_name: Name for cache file
            force_recompute: Force recomputation of cache
        """
        self.base_dataset = base_dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.cache_dir / f"{cache_name}.npz"
        self.force_recompute = force_recompute

        # Load or create cache
        self._load_or_create_cache()

    def _load_or_create_cache(self):
        """Load existing cache or create new one."""
        if self.cache_file.exists() and not self.force_recompute:
            logger.info(f"Loading cached features from {self.cache_file}")
            data = np.load(self.cache_file)
            self.cached_features = data['features']
            self.cached_labels = data['labels']
        else:
            logger.info(f"Creating cache at {self.cache_file}")
            self._create_cache()

    def _create_cache(self):
        """Create cache by processing all samples."""
        features = []
        labels = []

        for i in range(len(self.base_dataset)):
            if i % 100 == 0:
                logger.info(f"Caching sample {i}/{len(self.base_dataset)}")

            sample, label = self.base_dataset[i]

            # Convert tensor to numpy for storage
            if isinstance(sample, torch.Tensor):
                sample = sample.numpy()

            features.append(sample)
            labels.append(label)

        self.cached_features = np.array(features)
        self.cached_labels = np.array(labels)

        # Save to disk
        np.savez_compressed(
            self.cache_file,
            features=self.cached_features,
            labels=self.cached_labels
        )
        logger.info(f"Cached {len(features)} samples to {self.cache_file}")

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.cached_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get cached sample."""
        feature = torch.from_numpy(self.cached_features[idx]).float()
        label = int(self.cached_labels[idx])
        return feature, label