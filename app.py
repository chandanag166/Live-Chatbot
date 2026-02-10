from flask import Flask, render_template, request, jsonify
from chatbot import predict_class, get_response
from datetime import datetime
import re

app = Flask(__name__)

# Memory
user_context = {}
user_name = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_msg = request.form["msg"].strip()
    user_id = "default_user"

    tag = predict_class(user_msg)
    prev_intent = user_context.get(user_id)
    name = user_name.get(user_id)

    response = ""

    # -----------------------------
    # NAME CAPTURE LOGIC (FIXED)
    # -----------------------------
    if prev_intent == "ask_name":
        # Case 1: user enters just name (single word)
        if user_msg.isalpha() and len(user_msg) <= 20:
            name = user_msg.capitalize()
            user_name[user_id] = name
            response = f"Nice to meet you, {name} 😊 How can I help you today?"
        else:
            response = "Sorry, I didn't catch your name. Could you repeat it?"

    # Case 2: user gives name in sentence
    elif tag == "tell_name":
        match = re.search(
            r"(my name is|i am|call me|this is)\s+([A-Za-z]+)",
            user_msg.lower()
        )
        if match:
            name = match.group(2).capitalize()
            user_name[user_id] = name
            response = f"Nice to meet you, {name} 😊 How can I help you today?"
        else:
            response = "Nice to meet you! How can I help you today?"

    # -----------------------------
    # CONTEXT LOGIC
    # -----------------------------
    elif prev_intent == "admission" and "cse" in user_msg.lower():
        response = f"CSE admissions start in June, {name}." if name else "CSE admissions start in June."

    else:
        response = get_response(tag)
        if name:
            response = f"{response} {name} 😊"

    user_context[user_id] = tag

    # Log
    with open("logs/chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | User: {user_msg} | Bot: {response}\n")

    return jsonify({
        "response": response,
        "time": datetime.now().strftime("%H:%M")
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

