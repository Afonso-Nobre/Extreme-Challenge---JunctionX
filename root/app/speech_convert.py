"""
Transcription script using OpenAI Whisper and FFmpeg.

Dependencies:
- whisper (https://github.com/openai/whisper) - MIT License
- FFmpeg (https://ffmpeg.org/) - LGPL/GPL License

This script converts audio/video files to text and optionally returns
chunks of words with approximate timestamps.

Created with the help of OpenAI ChatGPT.
"""

import whisper
from pathlib import Path

model = whisper.load_model("base")



def convert(filename: str, words_per_chunk: int = 100):
    """
    Converts a file (mp3 or mp4, other types also supported, as long ffmpeg supports them)
    :param words_per_chunk:
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

    word_table = []
    for segment in result["segments"]:
        words = segment["text"].strip().split()
        start_time = segment["start"]

        for i in range(0, len(words), words_per_chunk):
            chunk = words[i : i + words_per_chunk]
            chunk = " ".join(chunk)
            word_table.append((chunk, start_time))

    return word_table


#table = convert("Power_English_Update.mp3")
#for chunk, start_time in table[:10]:
#    print(chunk, start_time)