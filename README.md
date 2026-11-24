# BMO
Creating my own BMO inspired by adventure time

Program structure
BMO has six states currently, this is done to better differentiate between when BMO is supposed to listen or act

Sleep:
BMO doesnt actively listen/record, he takes chunks of audio and looks if his wakeword has been said (Hey BMO)

Listening:
BMO starts actively listening/recording. If he finds a prolonged period of silence he stops and moves on.

Thinking:
BMO transcribes the audio he recorded and hands it over to his ollama model. If a tool is needed he will call the tool.

Speaking:
BMO speaks his answer.

Noisy:
When BMO is asked to perform a task like playing music or a video

Bored:
BMO has been left alone for a period of time and thus is bored and begins to watch videos, listen to music or play games.

Conversation loop:
- waits for wakeword and counts down until BMO gets bored
  - when bored will do random actions
- listens
  - when only silence goes back to sleep
- transcribes and thinks
  - if told to shut down he terminate
  - if told goodbye he will go back to sleep
  - calls tool if needed
- speaks
- waits to see if theres more to listen to otherwise goes to sleep
