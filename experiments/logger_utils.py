"""
Logging utilities for structured experiment logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_experiment_logging(
    log_dir: Path,
    experiment_name: str,
    level: str = "INFO",
    console_output: bool = True
) -> logging.Logger:
    """
    Setup structured logging for experiments.

    Args:
        log_dir: Directory for log files
        experiment_name: Name of the experiment
        level: Logging level
        console_output: Whether to output to console

    Returns:
        Configured logger
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # File handler for detailed logs
    file_handler = logging.FileHandler(log_dir / f"{experiment_name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler for important messages
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    return logger


class MetricsLogger:
    """Simple metrics logger that writes to JSON and CSV files."""

    def __init__(self, log_dir: Path, experiment_name: str):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.json"
        self.csv_file = self.log_dir / f"{experiment_name}_metrics.csv"

        self.metrics = []
        self._csv_header_written = False

    def log(self, metrics: dict, step: Optional[int] = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        # Add metadata
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            **metrics
        }

        self.metrics.append(entry)

        # Write to JSON file
        import json
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Write to CSV file
        self._write_csv_entry(entry)

    def _write_csv_entry(self, entry: dict):
        """Write single entry to CSV file."""
        import csv

        # Get all keys for header
        if not self._csv_header_written:
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=entry.keys())
                writer.writeheader()
                self._csv_header_written = True

        # Append data
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            writer.writerow(entry)


class ProgressLogger:
    """Simple progress logger for training loops."""

    def __init__(self, logger: logging.Logger, total_steps: int, log_frequency: int = 10):
        """
        Initialize progress logger.

        Args:
            logger: Logger instance
            total_steps: Total number of steps
            log_frequency: How often to log progress
        """
        self.logger = logger
        self.total_steps = total_steps
        self.log_frequency = log_frequency
        self.start_time = datetime.now()

    def log_step(self, step: int, metrics: dict):
        """
        Log progress for a step.

        Args:
            step: Current step number
            metrics: Metrics dictionary
        """
        if step % self.log_frequency == 0 or step == self.total_steps:
            # Calculate progress
            progress = (step / self.total_steps) * 100
            elapsed = datetime.now() - self.start_time
            eta = elapsed * (self.total_steps - step) / step if step > 0 else "Unknown"

            # Format metrics
            metrics_str = " | ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in metrics.items()])

            # Log message
            self.logger.info(
                f"Step {step:4d}/{self.total_steps} ({progress:5.1f}%) | "
                f"Elapsed: {elapsed} | ETA: {eta} | {metrics_str}"
            )

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: Optional[dict] = None):
        """
        Log epoch summary.

        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
        """
        # Format training metrics
        train_str = " | ".join([f"train_{k}={v:.4f}" if isinstance(v, float) else f"train_{k}={v}"
                               for k, v in train_metrics.items()])

        message = f"Epoch {epoch:3d} | {train_str}"

        # Add validation metrics if available
        if val_metrics:
            val_str = " | ".join([f"val_{k}={v:.4f}" if isinstance(v, float) else f"val_{k}={v}"
                                 for k, v in val_metrics.items()])
            message += f" | {val_str}"

        self.logger.info(message)


def create_experiment_loggers(config, experiment_name: str):
    """
    Create all necessary loggers for an experiment.

    Args:
        config: Configuration object
        experiment_name: Name of the experiment

    Returns:
        Tuple of (main_logger, metrics_logger)
    """
    log_dir = Path(config.logging.log_dir) / experiment_name

    # Setup main logger
    main_logger = setup_experiment_logging(
        log_dir=log_dir,
        experiment_name=experiment_name,
        level=config.logging.level,
        console_output=True
    )

    # Setup metrics logger
    metrics_logger = MetricsLogger(
        log_dir=log_dir,
        experiment_name=experiment_name
    )

    main_logger.info(f"Initialized loggers for experiment: {experiment_name}")
    main_logger.info(f"Log directory: {log_dir}")

    return main_logger, metrics_logger