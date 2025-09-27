import torch
import numpy as np

def load_model(model_path: str):
    """
    Load a trained model for inference.

    Args:
        model_path (str): Path to the trained model file.

    Returns:
        torch.nn.Module: Loaded model.
    """
    model = torch.load(model_path)
    model.eval()
    return model

def infer(model, features: np.ndarray):
    """
    Perform inference using the loaded model.

    Args:
        model (torch.nn.Module): Loaded model.
        features (np.ndarray): Input features for inference.

    Returns:
        np.ndarray: Model predictions.
    """
    with torch.no_grad():
        inputs = torch.from_numpy(features).float()
        outputs = model(inputs)
    return outputs.numpy()