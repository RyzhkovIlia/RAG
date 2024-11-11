import subprocess

command = "bash standalone_embed.sh start"
result = subprocess.run(command, shell=True, capture_output=True, text=True, executable="/bin/bash")

if result.returncode == 0:
    print("Command output:\n", result.stdout)
else:
    print("Error:\n", result.stderr)

from flask import Flask, request, render_template_string
import google.generativeai as genai

from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from get_answer import generate_answer_prompt
from src.tools import get_hybrid_function, get_gpt_model

# Loading the GPT and Milvus model
name_of_collection = 'Tresis'
retriever = get_hybrid_function(
    text_field='text', 
    name_of_collection=name_of_collection
)
model = get_gpt_model()

# Creating a Web page
app = Flask(__name__)

# HTML template for a page with an input field
html_template = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System</title>
</head>
<body>
    <h1>What do you want to know?:</h1>
    <form action="/" method="post">
        <textarea name="user_text" rows="4" cols="50" placeholder="Your question..."></textarea><br><br>
        <input type="submit" value="Send">
    </form>
    
    {% if processed_text %}
        <h2>Answer:</h2>
        <p>{{ processed_text }}</p>
    {% endif %}
</body>
</html>
"""

# Text processing function
def process_text(
        text:str, 
        retriever:MilvusCollectionHybridSearchRetriever, 
        model:genai.GenerativeModel
    ) -> str:
    answer = generate_answer_prompt(query=text, retriever=retriever, model=model)

    return answer

# Route for the home page
@app.route("/", methods=["GET", "POST"])
def index():
    processed_text = None
    if request.method == "POST":
        user_text = request.form["user_text"]
        processed_text = process_text(text=user_text, retriever=retriever, model=model)
    return render_template_string(html_template, processed_text=processed_text)

if __name__ == "__main__":
    app.run(debug=True)
