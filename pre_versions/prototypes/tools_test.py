from ollama import chat

MODEL = "llama3.2"

def get_time() -> str:
    """Get the current time."""
    return "It is 14:32."

def take_photo() -> str:
    """Take a photo with the camera."""
    return "Photo taken: photo_001.jpg"

TOOLS = {f.__name__: f for f in (get_time, take_photo)}

messages = [{
    "role": "system",
    "content": "You are a helpful assistant. Use a tool only when the user clearly asks for what it does.",
}]

try:
    while True:
        prompt = input("> ")
        messages.append({"role": "user", "content": prompt})

        for _ in range(3):                                   # cap on tool rounds
            response = chat(model=MODEL, messages=messages, tools=list(TOOLS.values()))
            msg = response.message
            messages.append(msg)                             # keeps any tool calls in the history

            if not msg.tool_calls:                           # a normal answer: print and finish
                print(msg.content)
                break

            for call in msg.tool_calls:                      # the model asked for tools
                name = call.function.name
                func = TOOLS.get(name)
                result = func(**call.function.arguments) if func else f"Unknown tool: {name}"
                print(f"[tool] {name} -> {result}")          # shows you that it really ran
                messages.append({"role": "tool", "tool_name": name, "content": str(result)})
except KeyboardInterrupt:
    pass
