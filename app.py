
import os
from dotenv import load_dotenv, dotenv_values
import markdown
from langchain_google_genai import ChatGoogleGenerativeAI
from flask import Flask,  render_template, request, jsonify

load_dotenv()
print(os.getenv("GOOGLE_API_KEY"))
app = Flask(__name__)


def LLM_Process(user_query):

    api_key = os.getenv("GOOGLE_API_KEY")
    

    print(api_key)

    if api_key is None:
        print("LLM API Key is not set or expired.")
        exit(0)

    

    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7,max_tokens=10000, timeout=None,max_retries=2)

    # messages = [("System","Hi, Welcome to your personal AI Chatbot"),("human", "I love Reading books"),]

    messages = [
        (
            "system",
            "You are a helpful assistant that answers to user queries in structured and clear manner which provides responses from reliable resources.",
        ),
        ("human", user_query),
    ]

    ai_msg = llm.invoke(messages)
    response_text = ai_msg.content or "No Response Received, Please report this incident."
    tokens = llm.max_output_tokens
    print("Output Tokens: ", tokens)
    response_html = markdown.markdown(response_text, extensions=["fenced_code", "tables"])
    return response_html


@app.route('/')
def index():
  
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
   
    user_input = request.json.get('query')
    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    response_html = LLM_Process(user_input)
    return jsonify({"response": response_html})

if __name__ == '__main__':
    # This is for local development only. App Engine uses the 'entrypoint' from app.yaml.
    # app.run(debug=True, host='0.0.0.0', port=8080)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


