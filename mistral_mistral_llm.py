from mistralai.client import Mistral
from dotenv import load_dotenv
import os

load_dotenv()



client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

'''
response = client.chat.complete(
    model="mistral-large-latest",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is my name ?"}
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.choices[0].message.content)
print(f"\nTokens used: {response.usage.total_tokens}")
'''


conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("🤖 Mistral  Chatbot (type 'quit' to exit)\n")

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
    stream = client.chat.stream(                    ## Different from other llm ,like openAI
        model="mistral-large-latest",
        messages=conversation_history,
        temperature=0.7,
        max_tokens=1000
    )

    print("Bot: ", end="", flush=True)
    full_response = ""

    for chunk in stream:
        content = chunk.data.choices[0].delta.content  ## Different from other llm . openAI
        if content:
            print(content, end="", flush=True)
            full_response += content

    print("\n")

    # Add bot response to history so it remembers context
    conversation_history.append({"role": "assistant", "content": full_response})