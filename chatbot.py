from flask import Flask, request, jsonify
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

app = Flask(__name__)

# 1. Load the Open-Source LLM and Tokenizer
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # change if OOM
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,   # changed from bfloat16
    device_map="auto",
)

# Create a Hugging Face pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512
)

llm = HuggingFacePipeline(pipeline=pipe)

# 2. Create a Prompt Template
template = """You are a helpful and friendly chatbot. Answer user questions clearly and concisely.

Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# 3. Create an LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "Question not provided"}), 400

    try:
        response = llm_chain.run(question)
        return jsonify({"response": response.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
