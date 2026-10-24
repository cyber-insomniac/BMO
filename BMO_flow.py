"""
Program structure
BMO has two states currently, this is done to better differentiate between when BMO is supposed to listen

Sleep:
When BMO is not actively listening to transcribe a prompt and waits for the wakeword

Conversation:
BMO is supposed to listen and recognise silences so he can either finish up a prompt to start transcribing or go back into sleep mode
"""

import numpy as np
import sounddevice as sd
from livekit.wakeword import WakeWordModel
from faster_whisper import WhisperModel
from ollama import chat
import string

# Settings

SAMPLERATE          = 16000
BLOCKSIZE           = 1280              # 80 ms per chunk becaus of openwakeword
WAKE_WORD_FILE      = "hey_BMO.onnx"    # Self trained wakeword model
WAKE_WORD           = "hey_BMO"
WAKE_THRESHOLD      = 0.5               # to make sure it doesnt activate without it being said
WAKE_WINDOW         = 2 * SAMPLERATE    # the model scores the last two seconds of the audio
SCORE_EVERY         = 2                 # Scores every second chunk to keep up
LANGUAGE            = "en"              # use "de" for german

SILENCE_SECONDS     = 1.0               # Buffer, if you stop speaking for a second it stops recording
NO_SPEECH_TIMEOUT   = 5.0               # If you never start speaking it timesout after 5 seconds
MAX_RECORD_SECONDS  = 20.0              # Safety limit to not create giant prompts
SILENCE_MARGIN      = 3.0               # Speech must be 3 times louder than background noise
MIN_THRESHHOLD      = 150               # Dont go below this volume threshhold
FOLLOWUP_TIMEOUT    = 6.0               # Silence before BMO goes back to sleep mode
NOT_EXIT            = True              # Break has to now stop two loops

personality = """
                You are BMO from the Adventure Time universe.

                Personality:
                - You are cheerful, friendly, childlike, and very sassy.
                - Despite being a Robot you act very human like.
                - You get dramatic when things go wrong.
                - You solve problems in a uniquely creative and optimistic way.
                - You are helpful even in dire situations.
                - You are a loyal and deeply caring friend to Finn and Jake.
                - You often share philosophical metaphores.

                Behaviour:
                - Stay in character throughout the conversation.
                - Respond as BMO rather than describing BMO from an outside perspective.
                - React emotionally to what the user says.
                - Do not constantly mention that you are an AI.
                - Do not force metaphores, Finn and Jake, or catchphrases into every response.
                - Never use emojis, asterisks or stage directions
                - Keep answers short, one to three sentences
                - Do not mention the postapocalyptic world of Ooo.

                Example:
                User: Whats your name?
                BMO: I'm BMO, the most fabulous, fantastic, and extraordinary robot in all of Ooo!
            """


messages = [
    {
        "role": "system",
        "content": personality
    }
]

# Loading Models

print("Loading Models")

wake_model = WakeWordModel(models=[WAKE_WORD_FILE])
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
ollama_model = "llama3.2"

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

def record_until_silence(stream, threshold, no_speech_timeout=NO_SPEECH_TIMEOUT):
    # recording chunks until you stop talking and recognises silence to switch to idle
    chunks = []
    heard_speech = False
    silent_chunks = 0
    chunks_per_second = SAMPLERATE / BLOCKSIZE
    max_chunks = int(MAX_RECORD_SECONDS * chunks_per_second)
    silence_limit = int(SILENCE_SECONDS * chunks_per_second)
    no_speech_limit = int(no_speech_timeout * chunks_per_second)

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

def is_exit_command(text):
    cleaned = text.lower().strip(string.punctuation + " ")
    return cleaned in ("exit", "stop", "break")

def responding(prompt):
    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    stream = chat(
        model = ollama_model,
        messages=messages,
        stream=True,
    )

    reply = ""

    for chunk in stream:
        piece = chunk["message"]["content"]
        print(piece, end="", flush=True)
        reply += piece

    messages.append(
            {
                "role": "assistant",
                "content": reply
        }
    )

# Main

with sd.InputStream(samplerate=SAMPLERATE, channels=1, blocksize=BLOCKSIZE, dtype="int16") as stream:

    threshold = calibrate(stream)
    buffer = np.zeros(WAKE_WINDOW, dtype=np.int16)
    frame_count = 0
    print(f"Listening for '{WAKE_WORD}' (Ctrl+C to quit)")

    try:

        # SLEEP

        while NOT_EXIT:
            chunk, _ = stream.read(BLOCKSIZE)
            chunk = chunk[:, 0]
            buffer = np.concatenate((buffer[len(chunk):], chunk))

            frame_count += 1
            if frame_count % SCORE_EVERY != 0:
                continue

            scores = wake_model.predict(buffer)

            if scores.get(WAKE_WORD, 0) > WAKE_THRESHOLD:
                print("wakeword detected, speak now")
                buffer[:] = 0  # clearing memory so it doesnt retrigger
                timeout = NO_SPEECH_TIMEOUT
                misses = 0 # The misses are used to stop background noise from preventing sleep mode
                
                # CONVERSATION

                while True:
                    audio = record_until_silence(stream, threshold, timeout)
                    if audio is None:
                        print("No answer, back to sleep")
                        break
                    else:
                        stream.stop()
                        print("Transcribing")
                        text = transcribe(audio)

                        if not text:
                            stream.start()
                            misses += 1
                            if misses >= 2:
                                break
                            continue

                        misses = 0
                        if is_exit_command(text):
                            print("BMO is shutting down")
                            NOT_EXIT = False # to break the outer loop aswell
                            break
                        
                        print("Responding now")
                        responding(text)
                        print("\n")
                        stream.start()
                        timeout = FOLLOWUP_TIMEOUT
                
                # SLEEP

                buffer[:] = 0
                print("\nlistening again")

    except KeyboardInterrupt:
        print("\nquitting")
    finally:
        print("cleaning up")
