from settings import *
from tools import *

import numpy as np
import sounddevice as sd
from livekit.wakeword import WakeWordModel
from faster_whisper import WhisperModel
from ollama import chat
from ollama import ChatResponse
import string
from piper import PiperVoice, SynthesisConfig
import random
import threading
import queue
import traceback

class Bag:
    def __init__(self, items):
        self.items, self.bag = list(items), []
    def draw(self):
        if not self.bag:
            self.bag = random.sample(self.items, len(self.items))
        return self.bag.pop()

class BackEnd:
    def __init__(self, models:dict, personality:str, tools: dict=None):
        self._wake_model        = models["wake"]
        self._whisper_model     = models["whisper"]
        self._ollama_model      = models["ollama"]
        self._voice_model       = models["voice"]
        self._summary_model     = models["summary"]
        self._rate              = self._voice_model.config.sample_rate
        self._syn_config        = SynthesisConfig(length_scale=0.6)
        self._personality       = personality
        self._tools             = tools or {}
        self._summary           = ""
        self._dropped           = []
        self._messages          = [
            {
                "role": "system",
                "content": personality
            }
        ]
        self._last_change       = 0
        self._last_state        = None
        self._timer             = None
        self._phrases           = Bag(BORED_PHRASES)
        self._idle_actions      = {
            "talk"      : (5, self.idle_talk),
            "music"     : (3, self.idle_music),
            "video"     : (2, self.idle_video),
            "game"      : (1, self.idle_game),
            "nothing"   : (3, lambda: None)
        }
        self.state              = SLEEPING
        self.stop = threading.Event()
        self.speech_q = queue.Queue()
        self.audio_q = queue.Queue(maxsize=3)

    # funktions
    
    def patience(self):
        self._timer = random.randint(BORED_MIN, BORED_MAX)
        print("Timer: ", self._timer)

    def busy(self):
        return (media_playing()
                or game_running.is_set()
                or self.speech_q.unfinished_tasks > 0) 

    def speak_and_wait(self, text):
        self.say(text)
        self.speech_q.join()

    def gettin_bored(self):
        if self._timer is None:
            self.patience()
        current_time = time.monotonic()
        if self.state != BORED:
            if int(current_time) - int(self._last_change) >= self._timer:
                self.change_state(BORED)
                self.patience()
        else:
            self.bored()

    def bored(self):
        if self.busy():
            return
        names = list(self._idle_actions)
        weights = [self._idle_actions[n][0] for n in names]
        name = random.choices(names, weights=weights, k=1)[0]
        self._idle_actions[name][1]()
        music.started = videos.started = False 

    def idle_talk(self):
        self.speak_and_wait(self._phrases.draw())

    def idle_music(self):
        self.speak_and_wait("I am going to play some music!")
        music.play("random")

    def idle_video(self):
        self.speak_and_wait("Let's watch something!")
        videos.play("random")

    def idle_game(self):
        self.speak_and_wait("Look at me, I'm going to play a game!")
        game_running.set()
        game_requests.put(("pong", "bmo"))

    def change_state(self, state):
        if self._last_state == BORED and state == SLEEPING:
            return
        if self._last_state != state:
            print(state)
            self.state = state
            self._last_state = state
            self._last_change = time.monotonic()

    def rms(self, chunk):
        # Volume of an audio chunk in int16 samples
        x = chunk.astype(np.float32)
        x -= x.mean()
        return float(np.sqrt(np.mean(x ** 2)))

    def calibrate(self, stream, seconds=1.0):
        # Measuring the background noise so silence is relative to the room
        for _ in range(int(1.0 * SAMPLERATE / BLOCKSIZE)):
            stream.read(BLOCKSIZE)
        print("Calibrating background noise, stay quiet")
        levels = []
        for _ in range(int(seconds * SAMPLERATE / BLOCKSIZE)):
            chunk, _ = stream.read(BLOCKSIZE)
            levels.append(self.rms(chunk[:,0]))
        threshold = max(np.mean(levels) * SILENCE_MARGIN, MIN_THRESHHOLD)
        print(f"Speech threshold: {threshold:.0f}")
        return threshold

    def flush(self, stream):
        n = stream.read_available
        if n:
            stream.read(n)

    def record_until_silence(self, stream, threshold, no_speech_timeout=NO_SPEECH_TIMEOUT):
        # recording chunks until you stop talking and recognises silence to switch to sleep
        chunks = []
        heard_speech = False
        silent_chunks = 0
        chunks_per_second = SAMPLERATE / BLOCKSIZE
        max_chunks = int(MAX_RECORD_SECONDS * chunks_per_second)
        silence_limit = int(SILENCE_SECONDS * chunks_per_second)
        no_speech_limit = int(no_speech_timeout * chunks_per_second)

        for i in range(max_chunks):
            chunk, _ = stream.read(BLOCKSIZE)
            chunk = chunk[:,0]
            chunks.append(chunk)

            if self.rms(chunk) > threshold:
                heard_speech = True
                silent_chunks = 0
            else:
                silent_chunks += 1

            if heard_speech and silent_chunks >= silence_limit:
                break
            if not heard_speech and i >= no_speech_limit:
                return None

        return np.concatenate(chunks)

    def transcribe(self, audio_int16):
        audio = audio_int16.astype(np.float32)
        audio -= audio.mean()
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.7
        segments, info = self._whisper_model.transcribe(audio, language=LANGUAGE)
        return " ".join(segment.text for segment in segments).strip()

    def matches(self, text, phrases):
        cleaned = text.lower().strip(string.punctuation + " ")
        words = cleaned.split()

        return cleaned in phrases or (len(words) <= 4 and any(p in cleaned for p in phrases))

    def is_sleep_command(self, text):
        return self.matches(text, SLEEP_PHRASES)

    def is_shutdown_command(self, text):
        return self.matches(text, SHUTDOWN_PHRASES)

    def shutdown(self):
        self.say("Goodbye, my friend!")
        self.speech_q.join()
        self.stop.set()

    def say(self, text):
        text = clean_for_speech(text)
        if text:
            self.speech_q.put(text)

    def _history_chars(self):
        return sum(len(m.get("content") or "") for m in self._messages[1:])

    def trim_history(self):
        while len(self._messages) - 1 > MAX_MESSAGES or self._history_chars() > MAX_CHARS:
            nxt = next((i for i in range(2, len(self._messages))
                        if self._messages[i]["role"] == "user"), None)
            if nxt is None:
                break
            self._dropped.extend(self._messages[1:nxt])
            del self._messages[1:nxt]

    def update_summary(self):
        transcript = "\n".join(
                                f"{'User' if m['role'] == 'user' else 'BMO'}: {m['content']}"
                                for m in self._dropped
                                if m["role"] in ("user", "assistant") and m.get("content")
        )
        if not transcript:
            self._dropped = []
            return
        prompt = (
            "Below is a summary of an earlier conversation, followed by newer dialogue. "
            "Write an updated summary in at most 5 sentences. Keep only lasting facts: "
            "the user's name, preferences, plans and promises. Skip small talk.\n\n"
            f"Current summary: {self._summary or '(none)'}\n\nNew dialogue:\n{transcript}\n\nUpdated summary:"
        )
        try:
            r = chat(model=SUMMARY_MODEL, messages=[{"role": "user", "content": prompt}])
            self._summary = r.message.content.strip()
        except Exception:
            traceback.print_exc()
            return                                  
        self._dropped = []
        self._messages[0]["content"] = (
            self._personality + "\n\nWhat you remember from earlier:\n" + self._summary
        )

    def responding(self, prompt):
        self._messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        self.trim_history()

        full_reply = ""
        for _ in range(MAX_TOOL_ROUNDS):
            stream = chat(
                model=self._ollama_model,
                messages=self._messages,
                tools=list(self._tools.values()) or None,
                stream=True,
            )

            text, buf, tool_calls = "", "", []
            for chunk in stream:
                piece = chunk.message.content or ""
                text += piece
                buf += piece
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)

                while True:
                    m = SENTENCE_END.search(buf, MIN_SENTENCE)
                    if not m:
                        break
                    self.say(buf[:m.end()])
                    buf = buf[m.end():]
            if buf.strip():
                self.say(buf)

            msg = {"role": "assistant", "content": text}
            if tool_calls:
                msg["tool_calls"] = tool_calls
                print(msg["tool_calls"])
            self._messages.append(msg)
            full_reply += text

            if not tool_calls:
                break

            for call in tool_calls:
                name = call.function.name
                args = call.function.arguments or {}
                func = self._tools.get(name)
                try:
                    result = func(**args) if func else f"Unknown tool: {name}"
                except Exception as e:
                    traceback.print_exc()
                    result = f"Tool failed: {e}"
                print(f"[tool] {name}({args}) -> {result}")
                self._messages.append({"role": "tool", "tool_name": name, "content": str(result)})

        return full_reply

    # Main funktions

    def voice_loop(self):
        if self.stop.is_set():
            return
        try:
            with sd.InputStream(samplerate=SAMPLERATE, channels=1, blocksize=BLOCKSIZE, dtype="int16") as stream:
                threshold = self.calibrate(stream)
                buffer = np.zeros(WAKE_WINDOW, dtype=np.int16)
                frame_count = 0
                self.change_state(SLEEPING)

                while not self.stop.is_set():
                    # checking if its time to get bored
                    self.gettin_bored()
                    #SLEEP
                    self.change_state(NOISY) if media_playing() and self.state != BORED else self.change_state(SLEEPING)

                    chunk, _ = stream.read(BLOCKSIZE)
                    chunk = chunk[:,0]
                    buffer = np.concatenate((buffer[len(chunk):], chunk))

                    frame_count += 1
                    if frame_count % SCORE_EVERY != 0:
                        continue
                    scores = self._wake_model.predict(buffer)
                    if scores.get(WAKE_WORD, 0) <= WAKE_THRESHOLD:
                        continue

                    #AWAKE
                    music.duck()
                    videos.duck()
                    self.conversation(stream, threshold)
                    music.restore()
                    videos.restore()
                    self.change_state(NOISY) if media_playing() and self.state != BORED else self.change_state(SLEEPING)
                    self.update_summary
                    self.flush(stream)
                    buffer[:] = 0

        except Exception:
            traceback.print_exc()

    def conversation(self, stream, threshold):
        timeout = NO_SPEECH_TIMEOUT
        misses = 0

        while not self.stop.is_set():
            self.change_state(LISTENING)
            audio = self.record_until_silence(stream, threshold, timeout)
            if audio is None:
                return

            self.change_state(THINKING)
            text = self.transcribe(audio)
            if not text:
                misses += 1
                if misses >= 2:
                    return
                continue
            misses = 0

            if self.matches(text, STOP_MUSIC_PHRASES + STOP_VIDEO_PHRASES + STOP_GAME_PHRASES):
                music.stop()
                videos.stop()
                game_requests.put("quit")
                self.speech_q.join()
                return
            if self.is_shutdown_command(text):
                self.shutdown()
                return
            if self.is_sleep_command(text):
                return
            
            reply = self.responding(text)
            print("BMO:", reply)

            self.speech_q.join()
            if music.started or videos.started or game_started.is_set():
                music.started =  False
                videos.started = False
                game_started.clear()
                return
            self.flush(stream)
            timeout = FOLLOWUP_TIMEOUT

    def synth_loop(self):
        try:
            while not self.stop.is_set():
                try:
                    text = self.speech_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    parts = [c.audio_int16_array for c in self._voice_model.synthesize(text, syn_config=self._syn_config)]
                    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)
                except Exception:
                    traceback.print_exc()
                    audio = np.zeros(0, dtype=np.int16)
                self.audio_q.put(audio)
        except Exception:
            traceback.print_exc()

    def speech_loop(self):
        try:
            with sd.OutputStream(samplerate=self._rate, channels=1, dtype="int16") as out:
                while not self.stop.is_set():
                    try:
                        audio = self.audio_q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    try:
                        if len(audio):
                            self.change_state(SPEAKING)
                            out.write(audio.reshape(-1,1))
                    finally:
                        self.speech_q.task_done()
        except Exception:
            traceback.print_exc()


