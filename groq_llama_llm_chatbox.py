from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("🤖 Groq Chatbot (type 'quit' to exit)\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    if not user_input:
        continue

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_input})

    # Stream the response
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_history,
        temperature=0.7,
        max_tokens=1000,
        stream=True
    )

    print("Bot: ", end="", flush=True)
    full_response = ""

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            full_response += content

    print("\n")

    # Add bot response to history so it remembers context
    conversation_history.append({"role": "assistant", "content": full_response})
