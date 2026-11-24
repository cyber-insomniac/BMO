from settings import *
from backend import *
from frontend import *

import numpy as np
import sounddevice as sd
from livekit.wakeword import WakeWordModel
from faster_whisper import WhisperModel
from ollama import chat
from ollama import ChatResponse
import string
from piper import PiperVoice, SynthesisConfig
import threading, queue, traceback, time, re, random, pygame
from pathlib import Path
import re
import subprocess
import os
import json
import socket
import math
pygame.init()

class BMO:
    def __init__(self, models:dict, personality:str, eyes:dict, mouths:dict, tools: dict=None):
        self.BackEnd = BackEnd(models, personality, TOOLS)
        self.FrontEnd = FrontEnd(eyes, mouths)
        self.timer = BORED_TIMER

    def face_state(self):
        return self.BackEnd.state

    def run(self):
        if self.BackEnd.stop.is_set():
            return
        threading.Thread(target=self.BackEnd.voice_loop, daemon=True).start()
        threading.Thread(target=self.BackEnd.speech_loop, daemon=True).start()
        threading.Thread(target=self.BackEnd.synth_loop, daemon=True).start()

        self.FrontEnd.run(self.face_state, self.BackEnd.stop)

        self.BackEnd.stop.set()
        pygame.quit()

bmo = BMO(models, personality, eyes, mouths)

try:
    bmo.run()
finally:
    bmo.BackEnd.stop.set()
