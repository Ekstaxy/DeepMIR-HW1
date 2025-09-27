import os
import json
import numpy as np
import torch
import torchaudio
from source_separation import separate_audio_inference
from model_inference import load_model, infer
from torchaudio.transforms import Resample
import librosa

# Hardcoded artist_to_id mapping
artist_to_id = {
    'aerosmith': 0, 'beatles': 1, 'creedence_clearwater_revival': 2, 'cure': 3,
    'dave_matthews_band': 4, 'depeche_mode': 5, 'fleetwood_mac': 6, 'garth_brooks': 7,
    'green_day': 8, 'led_zeppelin': 9, 'madonna': 10, 'metallica': 11, 'prince': 12,
    'queen': 13, 'radiohead': 14, 'roxette': 15, 'steely_dan': 16, 'suzanne_vega': 17,
    'tori_amos': 18, 'u2': 19
}

# Create id_to_artist mapping
id_to_artist = {v: k for k, v in artist_to_id.items()}

def process_audio_chunks(waveform: np.ndarray, sample_rate: int, chunk_duration: float = 5.0, num_chunks: int = 5):
    """
    Process audio by removing silence and extracting random chunks.

    Args:
        audio_path (str): Path to the audio file.
        sample_rate (int): Target sample rate.
        chunk_duration (float): Duration of each chunk in seconds.
        num_chunks (int): Number of chunks to extract.

    Returns:
        List[np.ndarray]: List of audio chunks.
    """
    # Convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Remove silence
    audio = waveform.squeeze(0).numpy()
    non_silent_intervals = librosa.effects.split(audio, top_db=30)
    audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])

    # Extract random chunks
    chunk_size = int(sample_rate * chunk_duration)
    chunks = []
    for _ in range(num_chunks):
        if len(audio) <= chunk_size:
            start = 0
        else:
            start = np.random.randint(0, len(audio) - chunk_size + 1)
        chunk = audio[start:start + chunk_size]
        chunks.append(chunk)

    return chunks

def run_inference(data_dir: str, model_path: str, output_dir: str):
    """
    Run the complete inference pipeline.

    Args:
        data_dir (str): Path to the directory containing raw audio files.
        model_path (str): Path to the trained model file.
        output_dir (str): Directory to save the classification results.
    """
    # Load the model
    model = load_model(model_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Store predictions
    predictions = {}

    # Process each audio file in the directory
    for root, _, files in os.walk(data_dir):
        for idx, file in enumerate(files):
            if file.endswith(('.mp3', '.wav')):
                audio_path = os.path.join(root, file)

                # Step 1: Source Separation
                print(f"Processing file {idx+1}/{len(files)}: {file}")
                vocals_mono = separate_audio_inference(audio_path, output_dir, 10, 0.2)

                # Step 2: Silence Removal and Chunking
                print("Extracting chunks...")
                chunks = process_audio_chunks(vocals_mono, sample_rate=16000, chunk_duration=5.0, num_chunks=35)

                # Step 3: Model Inference
                print("Running model inference...")
                predictions_for_file = []
                for chunk in chunks:
                    chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
                    prediction = infer(model, chunk_tensor)
                    predictions_for_file.append(prediction)

                # Average predictions
                averaged_prediction = np.mean(predictions_for_file, axis=0)
                
                # Flatten the prediction if it's 2D (which it appears to be)
                if averaged_prediction.ndim > 1:
                    averaged_prediction = averaged_prediction.flatten()
                
                print(f"Final averaged_prediction shape: {averaged_prediction.shape}")
                print(f"Averaged prediction: {averaged_prediction}")

                # Get top-3 predictions
                top3_indices = np.argsort(averaged_prediction)[-3:][::-1]
                print(f"Top 3 indices: {top3_indices}")
                
                # Now the indices should be 1D and work properly
                top3_artists = [id_to_artist[int(idx)] for idx in top3_indices]

                # Use file id (without extension) as key
                file_id = os.path.splitext(file)[0]

                # Store prediction
                predictions[file_id] = top3_artists

    # Save all predictions to a single JSON file
    predictions_path = os.path.join(output_dir, "predictions.json")
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    # Example usage
    run_inference(
        data_dir="inference_pipeline/data",
        model_path="inference_pipeline/model/ckpt/ShortChunkCNNRes_ckpt_epoch_52.pt",  # Updated to point to the correct checkpoint file
        output_dir="inference_pipeline/results"
    )