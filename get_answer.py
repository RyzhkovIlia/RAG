import google.generativeai as genai
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from pymilvus import connections

# connect to Milvus
CONNECTION_URI = "http://localhost:19530"
connections.connect(uri=CONNECTION_URI)


# Create a prompt for the GPT model
def make_rag_prompt(
        query:str, 
        relevant_passage:str
    ) -> str:
    """
    Creates a prompt for a Retrieval-Augmented Generation (RAG) model, designed to answer questions using relevant text.

    Parameters:
    - query (str): The question that the assistant needs to answer.
    - relevant_passage (str): The text passage that contains relevant information for answering the query.

    Returns:
    - str: A formatted prompt for the assistant, including the query and relevant context, structured for a friendly, non-technical response in Russian.
    """
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""
    Human: You are a helpful and informative bot that answers questions using text from the passage below. 
    Be sure to answer in a full sentence, exhaustively, including all relevant background information.
    However, you are speaking to a non-technical audience, so be sure to break down complex concepts and
    keep your tone friendly and accommodating.
    If the passage is not relevant to the answer, you can ignore it.
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

# Getting a response from the GPT model
def generate_answer_prompt(
        query:str, 
        retriever:MilvusCollectionHybridSearchRetriever, 
        model:genai.GenerativeModel
    ) -> str:    
    """
    Generates an answer to a query by using a retrieval-based prompt and a generative model.

    Parameters:
    - query (str): The question that the assistant should answer.
    - retriever (MilvusCollectionHybridSearchRetriever): The retriever object used to fetch relevant documents for the query.
    - model (genai.GenerativeModel): The generative model used to generate the answer based on the prompt.

    Returns:
    - str: The model's generated answer in response to the query, based on the provided relevant context.
    """
    similar_docs = retriever.invoke(query)
    relevant_text = ". ".join([doc.page_content.capitalize() for doc in similar_docs])
    prompt = make_rag_prompt(query=query, relevant_passage=relevant_text)
    answer = model.generate_content(prompt)
    
    return answer.text