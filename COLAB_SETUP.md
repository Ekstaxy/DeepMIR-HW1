# Google Colab Setup Instructions

## Quick Start

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Enable GPU**:
   - Runtime → Change runtime type → Hardware accelerator → GPU → Save

3. **Setup Environment**: Run this in the first cell:

```python

# Run this in Google Colab first cell
import torch
import sys

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

# Install additional packages if needed
!pip install librosa soundfile wandb hydra-core omegaconf

# Mount Google Drive (optional, for saving results)
from google.colab import drive
drive.mount('/content/drive')

# Clone your repository
!git clone https://github.com/YOUR_USERNAME/DeepMIR-HW1.git
%cd DeepMIR-HW1

# Verify setup
from configs import load_experiment_config, setup_device
config = load_experiment_config("deep_learning", "baseline")
device = setup_device(config)
print(f"Ready to train on: {device}")

```

## Configuration for Colab

The project automatically detects Colab environment and adjusts:
- **Batch size**: Automatically reduced based on available GPU memory
- **Workers**: Set to 2 for optimal Colab performance
- **Device**: Auto-detects GPU availability

## Memory Management

For large datasets in Colab:
- Use `cache_features=False` to avoid memory issues
- Consider using smaller `max_duration` for audio (e.g., 20 seconds)
- Monitor GPU memory with `torch.cuda.memory_summary()`

## Saving Results

Save results to Google Drive:
```python
# In Colab, save to mounted drive
config.paths.checkpoints = "/content/drive/MyDrive/DeepMIR-HW1/checkpoints"
config.paths.predictions = "/content/drive/MyDrive/DeepMIR-HW1/predictions"
```

## Troubleshooting

- **Out of Memory**: Reduce batch_size in config
- **Slow Training**: Check if GPU is properly detected
- **Data Loading Issues**: Set num_workers=0 if multiprocessing fails
