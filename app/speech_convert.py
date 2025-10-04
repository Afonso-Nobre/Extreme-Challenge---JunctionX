import whisper
from pathlib import Path

model = whisper.load_model("base")



def convert(filename: str):
    """
    Converts a file (mp3 or mp4, other types also supported, as long ffmpeg supports them)
    :param filename: the name of the file to convert with an extension
    :return: nothing, new file with the name filename-trans.txt in directory data/transcripts
    """

    parent_dir = Path.cwd().parent
    data_path = parent_dir/"data"
    audio_path = data_path / "inputs" / filename

    transcript_dir = data_path / "transcripts"
    transcript_path = transcript_dir / f"{Path(filename).stem}-trans.txt"

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Converting {str(audio_path)} to {str(transcript_path)}")

    result = model.transcribe(str(audio_path))
    text = result["text"].strip()

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(text)

    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    print(f"Transcript file written to {str(transcript_path)}")

convert("test_bbc.mp3")