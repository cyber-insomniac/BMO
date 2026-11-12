import sounddevice as sd
from faster_whisper import WhisperModel

print("starting program")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

samplerate = 16000
channels = 1
duration = 5

print("starting recording")

myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
sd.wait()
audio = myrecording.flatten()

segments, info = model.transcribe(
    audio,
    language="en"

)

print("processing audio to text")

text = " ".join(segment.text for segment in segments)

print(text)
