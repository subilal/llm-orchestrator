from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

MAX_HISTORY = 10

print("🤖 DeepSeek Chatbot (type 'quit' to exit)\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    if not user_input:
        continue

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_input})

    # Trim history — keep system prompt + last 10 messages
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = [conversation_history[0]] + conversation_history[-MAX_HISTORY:]

    # Stream the response
    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=conversation_history,
        temperature=0.7,
        max_tokens=500,
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

    # Add bot response to history
    conversation_history.append({"role": "assistant", "content": full_response})