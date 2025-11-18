import os
import torchaudio
import torch
from demucs.pretrained import get_model
from demucs import apply
import noisereduce as nr
import numpy as np

# ----------------------------------------------------------
# CONFIG (change only these paths)
# ----------------------------------------------------------
INPUT_AUDIO = r"speech nonspeech\7.wav"
OUTPUT_DIR = r"speech nonspeech\separated"
MODEL_NAME = "htdemucs"
# ----------------------------------------------------------

def denoise_vocals(vocal_path):
    print("\n=== STEP 2: Removing noise from vocals.wav ===")
    audio, sr = torchaudio.load(vocal_path)

    # audio shape could be [channels, length] or [1, length]
    audio_np = audio.squeeze().numpy()  # remove extra dims

    # Apply noise reduction
    reduced = nr.reduce_noise(
        y=audio_np,
        sr=sr,
        prop_decrease=0.9,
        stationary=False
    )

    # Convert back to tensor: shape MUST be [channels, length]
    if reduced.ndim == 1:
        out_tensor = torch.from_numpy(reduced).unsqueeze(0)  # [1, length]
    else:
        out_tensor = torch.from_numpy(reduced)

    # Ensure 2D shape
    out_tensor = out_tensor.float()

    torchaudio.save(vocal_path, out_tensor, sr)
    print(f"✔ Noise removed and saved: {vocal_path}")
  

def separate_audio(input_audio, output_dir):
    if not os.path.isfile(input_audio):
        raise FileNotFoundError(f"Audio file not found: {input_audio}")

    os.makedirs(output_dir, exist_ok=True)

    print("Loading Demucs model:", MODEL_NAME)
    model = get_model(MODEL_NAME)
    model.cpu()
    model.eval()

    print("Loading audio...")
    wav, sr = torchaudio.load(input_audio)

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    wav = wav.unsqueeze(0)

    print("\n=== STEP 1: Source Separation ===")
    with torch.no_grad():
        sources = apply.apply_model(model, wav, device="cpu")[0]

    source_names = model.sources
    print("Sources:", source_names)

    saved_files = {}

    for i, name in enumerate(source_names):
        out_path = os.path.join(output_dir, f"{name}.wav")
        audio_to_save = sources[i]
        torchaudio.save(out_path, audio_to_save, sr)
        print(f"Saved: {out_path}")
        saved_files[name] = out_path

    # ----------------------------------------------------
    # Apply noise removal ONLY on vocals.wav
    # ----------------------------------------------------
    if "vocals" in saved_files:
        denoise_vocals(saved_files["vocals"])

    print("\n✔ DONE — Separation + Noise Removal Completed!")

# Run
separate_audio(INPUT_AUDIO, OUTPUT_DIR)
