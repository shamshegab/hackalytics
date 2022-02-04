from pydub import AudioSegment
import librosa
import soundfile as sf
import os
import io
from contextlib import redirect_stdout
import re

def fluency_detector(mp3_path):
    output_file = "temp_folder/fluency_ready.wav"
    converted_file = "fluency-files/fluency_result.wav"
# AudioSegment.converter = "audio-test/ffmpeg.exe"
# convert mp3 file to wav file
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(converted_file, format="wav")
    y, s = librosa.load(converted_file, sr=48000)
    sf.write(output_file, y, s, "PCM_24")

    myssp=__import__("my-voice-analysis")
    p="fluency_result" # Audio File title
    c=os.path.abspath("fluency-files/")
    f = io.StringIO()
    with redirect_stdout(f):
        myssp.mysppron(p,c)
    out = f.getvalue()
    numbers = re.findall(r'\d+', out)
    return numbers[0]
    