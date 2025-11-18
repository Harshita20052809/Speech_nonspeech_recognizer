import subprocess
import sys
import os
import re

# ----------- CONFIG -------------
BASE = r"speech nonspeech"

TRANSCRIPTION_FILE = os.path.join(BASE, "Speech text output", "transcription.txt")
NONSPEECH_FILE = os.path.join(BASE, "Speech text output", "nonspeech_report.txt")
FINAL_REPORT_DIR = os.path.join(BASE, "Final report")
FINAL_REPORT_FILE = os.path.join(FINAL_REPORT_DIR, "final_report.txt")
# -------------------------------- 


def run_script(script_name):
    script_path = os.path.join(BASE, script_name)
    
    if not os.path.isfile(script_path):
        print(f"❌ ERROR: Script not found: {script_path}")
        return
    
    print(f"\n============================")
    print(f"▶ RUNNING: {script_name}")
    print(f"============================\n")
    
    result = subprocess.run([sys.executable, script_path])
    
    if result.returncode == 0:
        print(f"\n✔ FINISHED: {script_name}\n")
    else:
        print(f"\n❌ FAILED: {script_name}\n")
        sys.exit(1)


# ------------------------------------
#  Extract SPEECH text (clean)
# ------------------------------------
def extract_speech():
    if not os.path.isfile(TRANSCRIPTION_FILE):
        return "❌ Speech transcription not found."

    with open(TRANSCRIPTION_FILE, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Clean: remove empty lines, unwanted spaces
    text = re.sub(r"\s+", " ", text)
    return text


# -----------------------------------------
#  Extract NON-SPEECH sound list (clean)
# -----------------------------------------
def extract_nonspeech():
    if not os.path.isfile(NONSPEECH_FILE):
        return "❌ Non-speech report not found."

    clean_lines = []

    with open(NONSPEECH_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip headings/empty lines
            if line == "" or line.startswith("NON-SPEECH") or line.startswith("="):
                continue

            # Only include lines with "Label: value"
            if ":" in line:
                clean_lines.append(line)

    if not clean_lines:
        return "No non-speech sounds detected."

    return "\n".join(clean_lines)


# -----------------------------------------
#       BUILD FINAL REPORT
# -----------------------------------------
def create_final_report():
    print("\n============================")
    print("📄 GENERATING FINAL REPORT")
    print("============================\n")

    os.makedirs(FINAL_REPORT_DIR, exist_ok=True)

    speech_text = extract_speech()
    nonspeech_text = extract_nonspeech()

    final_content = f"""
==============================
🔊 FINAL AUDIO ANALYSIS REPORT
==============================

--------------------------------------
🎤 CLEAN SPEECH TRANSCRIPTION
--------------------------------------
{speech_text}


--------------------------------------
🌫 DETECTED NON-SPEECH SOUNDS
--------------------------------------
{nonspeech_text}


======================================
✔ REPORT GENERATED SUCCESSFULLY
======================================
"""

    with open(FINAL_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(final_content)

    print(final_content)
    print("\n✔ Final report saved to:")
    print(FINAL_REPORT_FILE)


# -----------------------------------------
#                MAIN SEQUENCE
# -----------------------------------------
if __name__ == "__main__":

    
    run_script("Seperator.py")
    
    #run_script("denoise.py")
    
    run_script("speech_analyser.py")
    
    run_script("non-speech_analyser.py")

    create_final_report()

    print("\n======================================")
    print("🎉 ALL TASKS + CLEAN FINAL REPORT DONE")
    print("======================================")
