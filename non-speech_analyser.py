import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import pandas as pd
import os

# ========== CONFIG ==========
NON_SPEECH_AUDIO = r"separated\other.wav"

OUTPUT_FOLDER = r"Speech text output"
OUTPUT_REPORT = os.path.join(OUTPUT_FOLDER, "nonspeech_report.txt")

TOP_N = 10  # top predictions to include
# ============================

# Make sure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YAMNet model
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

# Load class names
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
class_names = pd.read_csv(class_map_path)['display_name'].tolist()

# Load NON-SPEECH audio
waveform, sr = librosa.load(NON_SPEECH_AUDIO, sr=16000)

# Run through YAMNet
scores, embeddings, spectrogram = yamnet(waveform)
scores = scores.numpy()

# Average scores
avg_scores = np.mean(scores, axis=0)

# Speech labels to ignore
speech_terms = ["Speech", "Conversation", "Narration", "Babbling"]

background_sounds = []

# Get top N non-speech predictions
top_indices = np.argsort(avg_scores)[::-1]

for i in top_indices:
    if len(background_sounds) >= TOP_N:
        break
    label = class_names[i]
    score = avg_scores[i]
    
    if not any(s in label for s in speech_terms):
        background_sounds.append((label, score))

# Build report text
report_text = "NON-SPEECH BACKGROUND SOUND REPORT\n"
report_text += "===================================\n\n"
for label, score in background_sounds:
    report_text += f"{label}: {score:.3f}\n"

# Write report to file
with open(OUTPUT_REPORT, "w") as f:
    f.write(report_text)

# Print the content of the report
print("\n--- REPORT CONTENT ---")
print(report_text)
print("--- END OF REPORT ---\n")
print("Report saved to:", OUTPUT_REPORT)