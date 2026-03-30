import multiprocessing

from flask import Flask, render_template, request, Response, stream_with_context
from openai import OpenAI
from mistralai.client import Mistral
from groq import Groq
from dotenv import load_dotenv
import json
import os

load_dotenv()

app = Flask(__name__)

# Clients
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MAX_HISTORY = 10

# Separate conversation history for each model
histories = {
    "mistral": [{"role": "system", "content": "You are a helpful assistant."}],
    "groq": [{"role": "system", "content": "You are a helpful assistant."}],
}



@app.route("/")
def index():
    return render_template("multi_chat.html")

@app.route("/chat/<model>", methods=["POST"])
def chat(model):
    if model not in histories:
        return {"error": "Unknown model"}, 400

    user_message = request.json.get("message", "").strip()
    if not user_message:
        return {"error": "Empty message"}, 400

    history = histories[model]
    history.append({"role": "user", "content": user_message})

    # Trim history
    if len(history) > MAX_HISTORY:
        histories[model] = [history[0]] + history[-MAX_HISTORY:]
        history = histories[model]

    def generate_mistral():
        full_response = ""
        stream = mistral_client.chat.stream(
            model="mistral-large-latest",
            messages=history,
            temperature=0.7,
            max_tokens=1000
        )
        for chunk in stream:
            content = chunk.data.choices[0].delta.content
            if content:
                full_response += content
                yield f"data: {json.dumps({'content': content})}\n\n"
        history.append({"role": "assistant", "content": full_response})
        yield f"data: {json.dumps({'done': True})}\n\n"

    def generate_groq():
        full_response = ""
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=history,
            temperature=0.7,
            max_tokens=1000,
            stream=True
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                yield f"data: {json.dumps({'content': content})}\n\n"
        history.append({"role": "assistant", "content": full_response})
        yield f"data: {json.dumps({'done': True})}\n\n"

    if model == "mistral":
        return Response(stream_with_context(generate_mistral()), mimetype="text/event-stream")
    else:
        return Response(stream_with_context(generate_groq()), mimetype="text/event-stream")


@app.route("/reset/<model>", methods=["POST"])
def reset(model):
    if model not in histories:
        return {"error": "Unknown model"}, 400
    histories[model] = [{"role": "system", "content": "You are a helpful assistant."}]
    return {"status": "reset"}


if __name__ == "__main__":
    app.run(debug=True)
