import pickle
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from pypdf import PdfReader
from typing import List
import re
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from pymilvus import (
    Collection,
    WeightedRanker
)
import google.generativeai as genai
import os


def load_pdf_file(file_path: str) -> str:
    """
    Loads text from a PDF file and returns it as a single string.

    Parameters:
    - file_path (str): Path to the PDF file.

    Returns:
    - str: The extracted text from the PDF, with line breaks replaced by spaces.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text.replace('\n', ' ')


def split_text_to_token(text: str) -> List[str]:
    """
    Splits a given text into smaller chunks based on sentences and token limits.

    Parameters:
    - text (str): The full text to be split.

    Returns:
    - List[str]: A list of text chunks, each within the token limit.
    """
    split_text = re.split('\\. ', text)
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = []
    for text in [i for i in split_text if i != ""]:
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name:str="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents):
        return self.model.encode(documents)

    def embed_query(self, query):
        return self.model.encode([query])[0]
    def save(self, path:str):
        self.model.save(path)
    

def save_params_for_retriever(
    collection: Collection, 
    dense_field: str,
    sparse_field: str,
    dense_embedding_func: SentenceTransformerEmbeddings,
    sparse_embedding_func: BM25SparseEmbedding
    ) -> None:
    dense_embedding_func.save(f'src/embed_models/dense_embedding_model_{collection.name}')
    config = {
        'colection_name': collection.name,
        'dense_field': dense_field,
        'sparse_field': sparse_field,
        'sparse_embedding_func': sparse_embedding_func
    }

    with open(f"src/configs/config_params_{collection.name}.pkl", "wb") as f:
        pickle.dump(config, f)


def get_hybrid_function(
        text_field: str, 
        name_of_collection: str,
        k: int=5, 
        metric: str='IP'
    ) -> MilvusCollectionHybridSearchRetriever:

    with open(f'src/configs/config_params_{name_of_collection}.pkl', "rb") as f:
        config = pickle.load(f)
    model = SentenceTransformerEmbeddings(f'src/embed_models/dense_embedding_model_{name_of_collection}')
    sparse_search_params = {"metric_type": metric}
    dense_search_params = {"metric_type": metric, "params": {}}
    retriever = MilvusCollectionHybridSearchRetriever(
        collection = Collection(name_of_collection),
        rerank=WeightedRanker(0.5, 0.5),
        anns_fields=[config['dense_field'], config['sparse_field']],
        field_embeddings=[model, config['sparse_embedding_func']],
        field_search_params=[dense_search_params, sparse_search_params],
        top_k=k,
        text_field=text_field,
        nprobe=20
    )

    return retriever

def get_gpt_model() -> genai.GenerativeModel:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')

    return model