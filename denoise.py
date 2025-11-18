import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from scipy.signal import wiener

INPUT_FILE = r"separated\vocals.wav"
OUTPUT_FILE = r"separated\vocals_clean.wav"

print("Loading vocals.wav ...")
audio, sr = librosa.load(INPUT_FILE, sr=None)

# ---------------------------------------------------
# STEP 1 — BASIC NOISE REDUCTION (Old Compatible Mode)
# ---------------------------------------------------
print("Applying spectral gating noise reduction...")

# Only valid arguments for old versions:
reduced_noise = nr.reduce_noise(
    y=audio,
    sr=sr
)

# ---------------------------------------------------
# STEP 2 — EXTRA CLEANUP (Wiener Filter)
# ---------------------------------------------------
print("Applying Wiener filter...")
cleaned = wiener(reduced_noise)

# Normalize
cleaned = cleaned / np.max(np.abs(cleaned))

print("Saving cleaned file:", OUTPUT_FILE)
sf.write(OUTPUT_FILE, cleaned, sr)

print("\n✔ DONE — Clean vocal saved as:", OUTPUT_FILE)
