# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a singer classification project using the Artist20 dataset with 20 artists. The project implements both traditional machine learning and deep learning approaches for music information retrieval (MIR).

## Key Commands

### Environment Setup
```bash
# Local development (CPU only)
pip install -r requirements.txt

# Google Colab (with GPU)
# See COLAB_SETUP.md for detailed instructions
```

### Evaluation and Testing
```bash
# Evaluate model predictions against ground truth
python hw1/count_score.py [answer_path] [prediction_path]

# Example
python hw1/count_score.py ./hw1/test_ans.json ./hw1/test_pred.json
```

## Data Architecture

### Dataset Structure
- **Training data**: Listed in `data/raw/artist20/train.json` - paths to MP3 files organized by artist/album
- **Validation data**: Listed in `data/raw/artist20/val.json` - same structure as training
- **Test data**: `data/raw/artist20/test/` contains 233 numbered MP3 files (001.mp3 to 233.mp3)
- **Audio format**: 16kHz mono MP3 files, full song length

### Critical Data Rules
- **NEVER** use test folder audio files for training/validation
- Use only `train.json` for training and `val.json` for validation
- Test data should only be used for final predictions

## Project Structure

The codebase follows a modular research project structure:

```
├── configs/          # Experiment configuration files (Hydra)
├── data/            # Dataset and preprocessing
├── models/          # Model implementations (traditional ML + deep learning)
├── experiments/     # Training and evaluation scripts
├── results/         # Outputs (checkpoints, logs, predictions, visualizations)
├── notebooks/       # Exploratory analysis
├── hw1/            # Original dataset and evaluation utilities
```

## Output Format Requirements

### Prediction Format
Test predictions must be in JSON format with top-3 predictions per sample:
```json
{
    "1": ["artist_name_1", "artist_name_2", "artist_name_3"],
    "2": ["artist_name_1", "artist_name_2", "artist_name_3"],
    ...
    "233": ["artist_name_1", "artist_name_2", "artist_name_3"]
}
```

### Evaluation Metrics
- Top-1 accuracy: First prediction matches ground truth
- Top-3 accuracy: Ground truth appears in top-3 predictions
- Final score: `top1_accuracy + 0.5 * top3_accuracy`

## Key Technologies

### Audio Processing
- **librosa**: Primary audio feature extraction library
- **torchaudio**: PyTorch audio processing
- **soundfile**: Audio file I/O

### Machine Learning Stack
- **Traditional ML**: scikit-learn (k-NN, SVM, Random Forest)
- **Deep Learning**: PyTorch for end-to-end models
- **Experiment Tracking**: Weights & Biases (wandb)
- **Configuration**: Hydra for experiment management

### Two-Task Architecture

**Task 1: Traditional ML**
- Extract hand-crafted audio features using librosa
- Apply preprocessing (standardization, normalization, pooling)
- Train classical ML models
- Report confusion matrix and accuracy metrics

**Task 2: Deep Learning**
- Design neural network architecture from scratch
- End-to-end training without pre-trained models
- Proper train/validation split (never use test data during development)
- Generate final test predictions

## Development Workflow

1. **Data Exploration**: Analyze Artist20 dataset characteristics
2. **Feature Engineering**: Extract meaningful audio features
3. **Baseline Implementation**: Simple models for both tasks
4. **Systematic Experimentation**: Track all experiments with wandb
5. **Model Iteration**: Improve based on validation results
6. **Final Evaluation**: Test best models once on test set

## Important Notes

- Random seeds must be set for reproducibility
- All experiments should be tracked and logged
- Code should be modular with clear separation of concerns
- Test set evaluation should happen only once at the end
- Both models must achieve reasonable validation accuracy
- Album-level train/validation split maintains 4:1:1 ratio (949/231/233 tracks)