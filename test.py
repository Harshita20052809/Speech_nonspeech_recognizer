import whisper

audio = r"separated\vocals.wav"

model = whisper.load_model("large-v3")  # MUST use large-v3

result = model.transcribe(
    audio,
    fp16=False,
    language="hi",
    task="transcribe",
    condition_on_previous_text=False,
    temperature=0,        # forces exact output
    no_speech_threshold=0.1
)

print(result["text"])
