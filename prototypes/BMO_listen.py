import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel

# Settings

SAMPLERATE           = 16000
BLOCKSIZE           = 1280          # 80 ms per chunk becaus of openwakeword
WAKE_WORD           = "hey_jarvis"  # temporary for now until I trained my own
WAKE_THRESHOLD      = 0.5           # to make sure it doesnt activate without it being said
LANGUAGE            = "en"          # use "de" for german

SILENCE_SECONDS     = 1.0           # Buffer, if you stop speaking for a second it stops recording
NO_SPEECH_TIMEOUT   = 5.0           # If you never start speaking it timesout after 5 seconds
MAX_RECORD_SECONDS  = 20.0          # Safety limit to not create giant prompts
SILENCE_MARGIN      = 3.0           # Speech must be 3 times louder than background noise
MIN_THRESHHOLD      = 150           # Dont go below this volume threshhold

# Loading Models

print("Loading Models")

openwakeword.utils.download_models()
wake_model = Model(wakeword_models=[WAKE_WORD])
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

# funktions

def rms(chunk):
    # Volume of on audio chunk in int16 samples
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

def calibrate(stream, seconds=1.0):
    # Measuring the background noise so silence is relative to the room
    print("Calibrating background noise, stay quiet")
    levels = []
    for _ in range(int(seconds * SAMPLERATE / BLOCKSIZE)):
        chunk, _ = stream.read(BLOCKSIZE)
        levels.append(rms(chunk[:,0]))
    threshold = max(np.mean(levels) * SILENCE_MARGIN, MIN_THRESHHOLD)
    print(f"Speech threshold: {threshold:.0f}")
    return threshold

def record_until_silence(stream, threshold):
    # recording chunks until you stop talking
    chunks = []
    heard_speech = False
    silent_chunks = 0
    chunks_per_second = SAMPLERATE / BLOCKSIZE
    max_chunks = int(MAX_RECORD_SECONDS * chunks_per_second)
    silence_limit = int(SILENCE_SECONDS * chunks_per_second)
    no_speech_limit = int(NO_SPEECH_TIMEOUT * chunks_per_second)

    for i in range(max_chunks):
        chunk, _ = stream.read(BLOCKSIZE)
        chunk = chunk[:, 0]
        chunks.append(chunk)

        if rms(chunk) > threshold:
            heard_speech = True
            silent_chunks = 0
        else:
            silent_chunks += 1

        if heard_speech and silent_chunks >= silence_limit:
            break
        if not heard_speech and i >= no_speech_limit:
            return None

    return np.concatenate(chunks)

def transcribe(audio_int16):
    audio = audio_int16.astype(np.float32) / 32768.0
    segments, info = whisper_model.transcribe(audio, language=LANGUAGE)
    return " ".join(segment.text for segment in segments).strip()

# Main

with sd.InputStream(samplerate=SAMPLERATE, channels=1, blocksize=BLOCKSIZE, dtype="int16") as stream:

    threshold = calibrate(stream)
    print(f"Listening for '{WAKE_WORD}' (Ctrl+C to quit)")

    try:
        while True:
            chunk, _ = stream.read(BLOCKSIZE)
            prediction = wake_model.predict(chunk[:,0])

            if prediction.get(WAKE_WORD, 0) > WAKE_THRESHOLD:
                print("wakeword detected, speak now")
                wake_model.reset()  # clearing memory so it doesnt retrigger

                audio = record_until_silence(stream, threshold)
                if audio is None:
                    print("didnt hear anything")
                else:
                    stream.stop()
                    print("Transcribing")
                    print("You said:", transcribe(audio))
                    stream.start()

                print("listening again")

    except KeyboardInterrupt:
        print("\nquitting")
    finally:
        print("cleaning up")
