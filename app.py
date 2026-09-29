import os
import json
from typing import TypedDict, Dict, Any
from flask import Flask, render_template, request, Response, jsonify
from langgraph.checkpoint.memory import MemorySaver
from layer4 import build_cbt_graph      

app = Flask(__name__)

# Initialize memory checkpointer once
memory_checkpointer = MemorySaver()

# Build and compile graph ONLY ONCE with checkpointer attached
compiled_app = build_cbt_graph(checkpointer=memory_checkpointer)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    user_input = data.get("message", "").strip()
    session_id = data.get("session_id", "default_user")

    if not user_input:
        return jsonify({"error": "Message cannot be empty"}), 400

    def event_stream():
        config = {"configurable": {"thread_id": session_id}}
        
        # Match GraphState schema defined in layer4.py
        initial_state = {
            "query": user_input,
            "is_crisis": False,
            "category": None,
            "target_condition": None,
            "context": "",
            "response": ""
        }

        # Stream values directly from graph execution
        for event in compiled_app.stream(initial_state, config=config, stream_mode="values"):
            if "response" in event and event["response"]:
                yield event["response"]

    return Response(event_stream(), mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)