from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()  # automatically reads from .env

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)