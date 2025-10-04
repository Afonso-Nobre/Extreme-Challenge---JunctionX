import whisper
from pathlib import Path

model = whisper.load_model("small")

def convert(filename: str):
    data_path = Path("data")
    audio_path = data_path / "inputs" / filename
    print(audio_path)
    transcript_dir = data_path / "transcripts"
    transcript_path = transcript_dir / f"{Path(filename).stem}-trans.txt"
   # if not audio_path.exists():
    #    raise FileNotFoundError(f"Audio file not found: {audio_path}")
   # if not transcript_path.exists():
    #    raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    print(f"Converting {audio_path} to {transcript_path}")
    result = model.transcribe(str(audio_path))
    text = result["text"].strip()
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Transcript file written to {transcript_path}")

convert("proba.mp3")