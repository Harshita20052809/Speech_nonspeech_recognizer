import whisper
import os
from datetime import datetime

# Output folder
OUTPUT_FOLDER = r"Speech text output"

# Create folder if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load models once (fast)
model_hi = whisper.load_model("large")      # Hindi model
model_en = whisper.load_model("large")      # English model
lang_model = whisper.load_model("base")     # For detection

def transcribe_audio(path):

    # Step 1: Detect language
    detect = lang_model.transcribe(path)
    detected_lang = detect["language"]
    print("Detected Language:", detected_lang)

    # Step 2: Choose model
    if detected_lang in ["hi", "ur", "hr"]:
        print("Using Hindi model...")
        result = model_hi.transcribe(path, language="hi")
    


    elif detected_lang == "en":
        print("Using English model...")
        result = model_en.transcribe(path, language="en")

    else:
        print("Unknown language, using English as default...")
        result = model_en.transcribe(path)

    text_output = result["text"]

    # Step 3: Save output as .txt
    save_speech_text(text_output)

    return text_output


def save_speech_text(text):
    # Create unique filename
    filename = f"transcription.txt"
    save_path = os.path.join(OUTPUT_FOLDER, filename)

    # Write file
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\n✔ Speech-to-text saved at:\n{save_path}\n")


# -------------------------
# TEST
# -------------------------
audio_path = r"D:\programs\python\speech nonspeech\separated\vocals.wav"
final_text = transcribe_audio(audio_path)
print("\nFinal Text Output:\n", final_text)
