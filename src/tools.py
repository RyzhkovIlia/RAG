import os
import re
import pickle
from typing import List
from numpy import ndarray
from dotenv import load_dotenv

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from pymilvus import (
    Collection,
    WeightedRanker
)
from pdfminer.high_level import extract_text


def load_pdf_file(file_path: str, start_page: int, end_page: int) -> list:
    """
    Extracts and processes text from a specified range of pages in a PDF file.

    Parameters:
    - file_path (str): The path to the PDF file to be processed.
    - start_page (int): The page number where text extraction should begin.
    - end_page (int): The page number where text extraction should end.

    Returns:
    - list: A list of text segments (split by two newlines) extracted from the specified pages of the PDF. 
            The text is cleaned by removing content within parentheses and other formatting artifacts.
    """
    text = extract_text(file_path, page_numbers=list(range(start_page, end_page)))
    text = re.sub(r'\(.*?\)', '', text.replace('\x0c', ''), flags=re.DOTALL)
    text = text.split('\n\n')


    return text

def split_text_to_token(text_split: list) -> List[str]:
    """
    Splits a list of text segments into smaller chunks based on sentence boundaries and token limits.

    Parameters:
    - text_split (list): A list of text segments to be split into smaller chunks.

    Returns:
    - List[str]: A list of smaller text chunks, each containing no more than the defined token limit 
                 (256 tokens per chunk) with no overlapping tokens between chunks.
    """
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = []
    for text in [i for i in text_split if i != ""]:
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts

class SentenceTransformerEmbeddings(Embeddings):
    """
    A class for generating dense embeddings for documents and queries using a SentenceTransformer model.
    
    This class provides methods to embed documents and queries, making it suitable for use in information 
    retrieval tasks. It also includes functionality to save the embedding model to a specified path.

    Args:
        model_name (str, optional): The name of the SentenceTransformer model to use for embeddings.
            Defaults to "all-MiniLM-L6-v2".

    Attributes:
        model (SentenceTransformer): The SentenceTransformer model used for encoding documents and queries.
    """
    def __init__(self, model_name:str="all-MiniLM-L6-v2"):
        """
        Initializes the SentenceTransformerEmbeddings instance with a specified SentenceTransformer model.
        
        Args:
            model_name (str, optional): The name of the SentenceTransformer model to use. 
                Defaults to "all-MiniLM-L6-v2".
        """
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents:list) -> list:
        """
        Generates embeddings for a list of documents.

        Args:
            documents (list of str): A list of documents to be embedded.

        Returns:
            numpy.ndarray: An array of embeddings for the provided documents.
        """
        return self.model.encode(documents)

    def embed_query(self, query:str) -> ndarray:
        """
        Generates an embedding for a single query.

        Args:
            query (str): A query to be embedded.

        Returns:
            numpy.ndarray: An embedding for the provided query.
        """
        return self.model.encode([query])[0]
    
    def save(self, path:str) -> None:
        """
        Saves the SentenceTransformer model to the specified path.

        Args:
            path (str): The path where the model should be saved.
        Returns:
            None
        """
        self.model.save(path)
    

def save_params_for_retriever(
    collection: Collection, 
    dense_field: str,
    sparse_field: str,
    dense_embedding_func: SentenceTransformerEmbeddings,
    sparse_embedding_func: BM25SparseEmbedding
    ) -> None:
    """
    Saves the parameters for setting up a Milvus hybrid retriever to a configuration file.

    Args:
        collection (Collection): The Milvus collection for which retriever parameters are being saved.
        dense_field (str): The name of the field for dense embeddings within the collection.
        sparse_field (str): The name of the field for sparse embeddings within the collection.
        dense_embedding_func (SentenceTransformerEmbeddings): The embedding function for dense text representations.
        sparse_embedding_func (BM25SparseEmbedding): The embedding function for sparse text representations.

    Returns:
        None
    """
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
    """
    Creates and returns a MilvusCollectionHybridSearchRetriever instance, configured for hybrid search
    with both dense and sparse embeddings.

    Args:
        text_field (str): The name of the field containing text data in the Milvus collection.
        config_path (str): The path to the configuration file (in pickle format) containing collection 
            and embedding settings.
        k (int, optional): The number of top results to retrieve. Defaults to 5.
        metric (str, optional): The similarity metric to use for search (e.g., 'IP' for inner product). Defaults to 'IP'.

    Returns:
        MilvusCollectionHybridSearchRetriever: A configured hybrid search retriever for searching
            within the Milvus collection using both dense and sparse embeddings.
    """
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
    """
    Initializes and returns a Generative AI model from the genai library using the Gemini API.

    This function retrieves the Gemini API key from the environment variables, configures 
    the genai library, and initializes a generative model. If the API key is not provided, 
    it raises an error.

    Returns:
        genai.GenerativeModel: An instance of the generative model configured with the 'gemini-pro' model name.

    Raises:
        ValueError: If the Gemini API key is not set in the environment variables as GEMINI_API_KEY.
    """
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')

    return model