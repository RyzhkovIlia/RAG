import google.generativeai as genai
import chromadb
from typing import List


def get_relevant_vectors(query: str, database: chromadb.Collection, n_results: int) -> List[str]:
    """
    Retrieves relevant passages from the Chroma collection based on a query.

    Parameters:
    - query (str): The search query.
    - db (chromadb.Collection): The Chroma collection to search.
    - n_results (int): The number of relevant results to return.

    Returns:
    - List[str]: A list of the most relevant passages.
    """
    passage = database.query(query_texts=[query], n_results=n_results, include=['documents', 'embeddings'])

    return passage


def rag_prompt(query: str, relevant_passage: str) -> str:
    """
    Constructs a prompt for a RAG (Retrieval-Augmented Generation) system.

    Parameters:
    - query (str): The query or question.
    - relevant_passage (str): The relevant passage to assist in generating the response.

    Returns:
    - str: A structured prompt for RAG.
    """
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""You are an informative and approachable bot that uses the reference passage provided below to answer questions. \
        Please respond in complete sentences, covering all essential details and context. \
            Remember, you're speaking to a non-technical audience, so simplify complex ideas and maintain a warm, conversational style. \
                If the passage does not help answer the question, feel free to disregard it.
                
    QUESTION: '{query}'
    PASSAGE: '{escaped}'

    ANSWER:
    """

    return prompt


def generate_answer(prompt: str, api_gemino: str) -> str:
    """
    Generates an answer based on the provided prompt using a generative AI model.

    Parameters:
    - prompt (str): The prompt for the generative model.

    Returns:
    - str: The generated response text.
    """
    gemini_api_key = api_gemino#os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)

    return answer.text


def request_db(database: chromadb.Collection, query: str, api_gemino: str) -> str:
    """
    Generates an answer to a query using relevant text from the Chroma collection.

    Parameters:
    - db (chromadb.Collection): The Chroma collection to retrieve relevant passages.
    - query (str): The question or query to answer.

    Returns:
    - tuple: A tuple containing the answer text and the relevant passages.
    """
    relevant_text = get_relevant_vectors(query=query, database=database, n_results=5)
    prompt = rag_prompt(query=query, relevant_passage="".join(relevant_text['documents'][0]))
    answer = generate_answer(prompt=prompt, api_gemino=api_gemino)

    return answer
