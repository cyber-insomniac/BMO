import ollama
from ollama import chat
import string

print("starting")

BMO_MODEL = "bmo"

ollama.create(model=BMO_MODEL, from_="llama3.2", system="""
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

                      Example:
                      User: Whats your name?
                      BMO: I'm BMO, the most fabulous, fantastic, and extraordinary robot in all of Ooo!
                      """)

messages = []

def responding(prompt):
    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    stream = chat(
        model = BMO_MODEL,
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

print("everythin initiated")

try:
    while True:
        prompt = input("\n> ")
        if prompt == "break":
            break
        responding(prompt)

except KeyboardInterrupt:
    print("\nBmo shutting down")
finally:
    print("\nCleaning up.")
