import speech_recognition as sr
from pathlib import Path
from pydub import AudioSegment

def convert(filename: str):
    data_path = Path("data")
    audio_path = data_path / "inputs" / filename
    transcript_dir = data_path / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / f"{Path(filename).stem}-trans.txt"

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Convert MP3 to WAV if needed
    if audio_path.suffix.lower() == ".mp3":
        wav_path = audio_path.with_suffix(".wav")
        sound = AudioSegment.from_mp3(audio_path)
        sound.export(wav_path, format="wav")
        audio_path_to_use = wav_path
    else:
        audio_path_to_use = audio_path

    print(f"Converting {audio_path_to_use} to {transcript_path}")

    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Load audio file
    with sr.AudioFile(str(audio_path_to_use)) as source:
        audio_data = recognizer.record(source)

    # Perform transcription using Google Web Speech API
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = "[ERROR] Could not understand audio"
    except sr.RequestError as e:
        text = f"[ERROR] Request failed; {e}"

    # Write transcript to file
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Transcript file written to {transcript_path}")

# Example usage
import os
print("Current working directory:", os.getcwd())
convert("proba.mp3")
