import torch
import numpy as np
from model.model import ShortChunkCNN_Res

def load_model(model_path: str):
    """
    Load a trained ShortChunkCNN_Res model for inference.

    Args:
        model_path (str): Path to the trained model checkpoint file.

    Returns:
        torch.nn.Module: Loaded model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Debug: Print the keys in the checkpoint
    print("Checkpoint keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "Not a dictionary")

    # Extract configuration from the checkpoint
    config = checkpoint.get("config", {})
    # print("Model configuration:", config)

    # Initialize the ShortChunkCNN_Res model with the checkpoint configuration
    model = ShortChunkCNN_Res(
        n_channels=config.get("model", {}).get("n_channels", 128),
        sample_rate=config.get("audio", {}).get("sample_rate", 16000),
        n_fft=config.get("audio", {}).get("n_fft", 2048),
        hop_length=config.get("audio", {}).get("hop_length", 512),
        f_min=config.get("audio", {}).get("f_min", 80.0),
        f_max=config.get("audio", {}).get("f_max", 8000.0),
        n_mels=config.get("audio", {}).get("n_mels", 80),
        n_class=config.get("dataset", {}).get("num_classes", 20)
    )

    # Check if the checkpoint contains the model state dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("The checkpoint does not contain a valid model state dictionary.")

    model.to(device)
    model.eval()
    return model

def infer(model, features):
    """
    Perform inference using the loaded model.

    Args:
        model (torch.nn.Module): Loaded model.
        features (np.ndarray or torch.Tensor): Input features for inference.

    Returns:
        np.ndarray: Model predictions.
    """
    # Ensure features are a Tensor
    if not isinstance(features, torch.Tensor):
        features = torch.from_numpy(features).float()

    with torch.no_grad():
        outputs = model(features)
    return outputs.numpy()