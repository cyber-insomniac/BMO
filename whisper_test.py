import sounddevice as sd
from faster_whisper import WhisperModel

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

samplerate = 16000
channels = 1
duration = 5

myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
sd.wait()
audio = myrecording.flatten()

segments, info = model.transcribe(
    audio,
    language="de"

)

text = " ".join(segment.text for segment in segments)

print(text)
