from flask import Flask, request, render_template_string
from get_answer import generate_answer_promt
from src.tools import get_hybrid_function, get_gpt_model
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
import google.generativeai as genai

# Загружаем модель GPT и Milvus
name_of_collection = 'Pushkin'
retriever = get_hybrid_function(
    text_field='text', 
    name_of_collection = name_of_collection
)
model = get_gpt_model()

# Создаем Web страницу
app = Flask(__name__)

# HTML-шаблон для страницы с полем ввода
html_template = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Processor</title>
</head>
<body>
    <h1>Что ты хочешь спросить в базе данных:</h1>
    <form action="/" method="post">
        <textarea name="user_text" rows="4" cols="50" placeholder="Введите текст здесь..."></textarea><br><br>
        <input type="submit" value="Отправить">
    </form>
    
    {% if processed_text %}
        <h2>Ответ:</h2>
        <p>{{ processed_text }}</p>
    {% endif %}
</body>
</html>
"""

# Функция обработки текста
def process_text(
        text:str, 
        retriever:MilvusCollectionHybridSearchRetriever, 
        model:genai.GenerativeModel
    ) -> str:
    answer = generate_answer_promt(query=text, retriever=retriever, model=model)

    return answer

# Маршрут для главной страницы
@app.route("/", methods=["GET", "POST"])
def index():
    processed_text = None
    if request.method == "POST":
        user_text = request.form["user_text"]
        processed_text = process_text(text=user_text, retriever=retriever, model=model)
    return render_template_string(html_template, processed_text=processed_text)

if __name__ == "__main__":
    app.run(debug=True)
