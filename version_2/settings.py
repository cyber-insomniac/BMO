from pathlib import Path
import re
import pygame
from livekit.wakeword import WakeWordModel
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig
pygame.init()

BASE = Path(__file__).parent
ROOT = BASE.parent

SAMPLERATE          = 16000
BLOCKSIZE           = 1280                                  # 80 ms per chunk becaus of openwakeword
WAKE_WORD_FILE      = ROOT / "sound_models" / "hey_BMO.onnx"# Self trained wakeword model
WAKE_WORD           = "hey_BMO"
VOICE_FILE          = ROOT / "sound_models" / "en_US-amy-medium.onnx"
WAKE_THRESHOLD      = 0.5                                   # to make sure it doesnt activate without it being said
WAKE_WINDOW         = 2 * SAMPLERATE                        # the model scores the last two seconds of the audio
SCORE_EVERY         = 2                                     # Scores every second chunk to keep up
LANGUAGE            = "en"                                  # use "de" for german

SILENCE_SECONDS     = 1.0                                   # Buffer, if you stop speaking for a second it stops recording
NO_SPEECH_TIMEOUT   = 5.0                                   # If you never start speaking it timesout after 5 seconds
MAX_RECORD_SECONDS  = 20.0                                  # Safety limit to not create giant prompts
SILENCE_MARGIN      = 3.0                                   # Speech must be 3 times louder than background noise
MIN_THRESHHOLD      = 40                                    # Dont go below this volume threshhold
FOLLOWUP_TIMEOUT    = 6.0                                   # Silence before BMO goes back to sleep mode
NOT_EXIT            = True                                  # Break has to now stop two loops
SENTENCE_END        = re.compile(r'[.!?]+["\')\]]*\s+|\n+') # So BMO can say individual sentences
MIN_SENTENCE        = 12                                    # Minimum size for a sentence
MAX_TOOL_ROUNDS     = 3                                     # so ollama doesnt call tools in a loop
MAX_MESSAGES        = 16                                    # limiting memory
MAX_CHARS           = 6000                                  # for the summary
NORMAL_VOL          = 0.5
DUCK_VOL            = 0.12
VIDEO_DUCK          = 50
VIDEO_NORMAL        = 100
BORED_TIMER         = 10                                    # Debug timer for testing
BORED_MIN           = 180
BORED_MAX           = 300

EYES_PATH           = ROOT / "assets" / "eyes"             # Asset paths for BMOs face
MOUTH_PATH          = ROOT / "assets" / "mouth"
SONG_PATH           = ROOT / "music"
VIDEO_PATH          = ROOT / "videos"
WIDTH               = 1024                                  # Screen size for pygame
HEIGHT              = 600
FPS                 = 60
FONT                = pygame.font.Font(None, 72)
SLEEPING            = "sleeping"                            # BMOs states
LISTENING           = "listening"
THINKING            = "thinking"
SPEAKING            = "speaking"
BORED               = "bored"
NOISY               = "noisy"
SLEEP_PHRASES       = ("stop", "go to sleep", "good night", "goodnight")
SHUTDOWN_PHRASES    = ("shut down", "shutdown", "power off", "turn off", "exit", "kill yourself")
STOP_MUSIC_PHRASES  = ("stop the music", "stop music", "music off", "turn off the music", "turn the music off", "pause the music", "stop the radio", "turn off the radio", "radio off")
STOP_VIDEO_PHRASES  = ("stop the video", "stop video", "video off", "turn off the video", "close the video")
STOP_GAME_PHRASES   = ("stop game", "stop the game", "quit the game", "end the game", "game over")
FILE_NAMES          = {".wav", ".mp4"}
BORED_PHRASES       = ["Hello darkness my old friend",
                       "NOOOOO dont leave",
                       "bastard",
                       "HELP",
                       "Going to goon",
                       "bruthaaaa come back",
                       ]

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

models = {
    "wake"      : WakeWordModel(models=[str(WAKE_WORD_FILE)]),
    "whisper"   : WhisperModel("small", device="cpu", compute_type="int8"),
    "ollama"    : "llama3.2",
    "voice"     : PiperVoice.load(str(VOICE_FILE)),
    "summary"   : "llama3.2"
}

eyes = {
    SLEEPING    : "normal.png",
    LISTENING   : "listen.png",
    THINKING    : "think.png",
    SPEAKING    : "speak.png",
    BORED       : "bored.png",
    NOISY       : "bored.png"

}

mouths = {
    SLEEPING    : ["normal1.png", "normal2.png", "normal3.png"],
    LISTENING   : ["listen1.png"],
    THINKING    : ["think1.png"],
    SPEAKING    : ["speak1.png", "speak2.png", "speak3.png", "speak4.png", "speak5.png", "speak6.png"],
    BORED       : ["bored1.png", "bored2.png"],
    NOISY       : ["bored1.png", "bored2.png"]
}

