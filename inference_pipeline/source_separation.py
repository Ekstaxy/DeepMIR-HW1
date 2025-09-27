import os
import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Resample, Fade
from pathlib import Path

# Load pre-trained model
bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model().eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
sample_rate = bundle.sample_rate

def separate_audio_inference(audio_path: str, output_dir: str, segment: float, overlap: float) -> str:
    """
    Perform source separation on an audio file for inference.

    Args:
        audio_path (str): Path to the input audio file.
        output_dir (str): Directory to save the separated audio files.

    Returns:
        str: Path to the separated vocals file.
    """
    # Load and preprocess audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

    # Move to device
    waveform = waveform.to(device)

    # Perform source separation
    chuck_len = int(segment * sample_rate * (1 + overlap))
    start = 0
    end = chuck_len
    overlap_frame = int(overlap * sample_rate)
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frame), fade_shape='linear')
    # Batch size 1, 4 sources, 2 channels, time
    final = torch.zeros(1, 4, 2, waveform.size(1))

    while start < waveform.size(1)-overlap_frame:
        chunk = waveform[:, start:end]
        # Pad chunk if it's shorter than chuck_len
        if chunk.size(1) < chuck_len:
            padding = torch.zeros(waveform.size(0), chuck_len - chunk.size(1)).to(device)
            chunk = torch.cat((chunk, padding), dim=1)
        
        # Ensure chunk has 2 channels
        if chunk.size(0) == 1:
            chunk = chunk.repeat(2, 1)
        # Separate chunk
        with torch.inference_mode():
            separated_chunk = model(chunk.unsqueeze(0))

        separated_chunk = fade(separated_chunk)
        separated_chunk = separated_chunk[:, :, :, :end-start]
        final[:, :, :, start:end] += separated_chunk
        # Overlap-add
        if start == 0:
            fade.fade_in_len = int(overlap_frame)
            start += chuck_len - overlap_frame
        else:
            start += chuck_len
        
        end += chuck_len
        if end > waveform.size(1):
            end = waveform.size(1)
    
    source_names = ['drums', 'bass', 'other', 'vocals']
    vocals = final[0][3]


    vocals = Resample(orig_freq=sample_rate, new_freq=sr)(vocals)
    if vocals.dim() == 2 and vocals.size(0) == 2:
        vocals_mono = vocals.mean(dim=0, keepdim=True)
    else:
        vocals_mono = vocals

    return vocals_mono