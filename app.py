
import os
from dotenv import load_dotenv, dotenv_values

from langchain_google_genai import ChatGoogleGenerativeAI

from flask import Flask,  render_template, request, jsonify
load_dotenv()
print(os.getenv("GOOGLE_API_KEY"))
app = Flask(__name__)


def LLM_Process(user_query):

    api_key = os.getenv("GOOGLE_API_KEY")
    # api_key = "AIzaSyAq9D9SnvZa67Aoo41mdoz7ZyqASvSJrM4"

    print(api_key)

    if api_key is None:
        print("LLM API Key is not set or expired.")
        exit(0)

    

    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7,max_tokens=1000, timeout=None,max_retries=2)

    # messages = [("System","Hi, Welcome to your personal AI Chatbot"),("human", "I love Reading books"),]

    messages = [
        (
            "system",
            "You are a helpful assistant that answers to user queries in structured and clear manner which provides responses from reliable resources.",
        ),
        ("human", user_query),
    ]

    ai_msg = llm.invoke(messages)
    response = ai_msg.content
    return response


@app.route('/')
def index():
    """Serves the main interactive page."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handles user queries and returns AI responses."""
    user_input = request.json.get('query')
    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    ai_response = LLM_Process(user_input)
    return jsonify({"response": ai_response})

if __name__ == '__main__':
    # This is for local development only. App Engine uses the 'entrypoint' from app.yaml.
    # app.run(debug=True, host='0.0.0.0', port=8080)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)