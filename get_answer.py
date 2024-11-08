import google.generativeai as genai
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from pymilvus import (
    connections
)
CONNECTION_URI = "http://localhost:19530"
connections.connect(uri=CONNECTION_URI)


# Создаем промпт для модели GPT
def make_rag_prompt(
        query:str, 
        relevant_passage:str
    ) -> str:

    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""
    Human: You are a helpful and informative bot that answers questions using text from the passage below. 
    Be sure to answer in a full sentence, exhaustively, including all relevant background information.
    However, you are speaking to a non-technical audience, so be sure to break down complex concepts and
    keep your tone friendly and accommodating.
    If the passage is not relevant to the answer, you can ignore it.
    Please write in Russian. 
    Context is given between <context>.
    The question is given between </question>.
    Thank you!

    <context>
    {escaped}
    </context>

    <question>
    {query}
    </question>

    Assistant:
    """

    return prompt

# Получаем ответ от модели GPT
def generate_answer_promt(
        query:str, 
        retriever:MilvusCollectionHybridSearchRetriever, 
        model:genai.GenerativeModel
    ) -> str:    

    similar_docs = retriever.invoke(query)
    relevant_text = ". ".join([doc.page_content.capitalize() for doc in similar_docs])
    prompt = make_rag_prompt(query=query, relevant_passage=relevant_text)
    answer = model.generate_content(prompt)
    
    return answer.text