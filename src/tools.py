import re
from tqdm import tqdm
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from pypdf import PdfReader
import chromadb
import umap.umap_ as umap
from typing import List, Dict, Any
import numpy as np
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction

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
    split_text = re.split('\. ', text)
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = []
    for text in [i for i in split_text if i != ""]:
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts


def create_chroma_database(documents: List[str], name: str, embedding_function: SentenceTransformerEmbeddingFunction, path: str) -> chromadb.Collection:
    """
    Creates a Chroma database collection with the given documents.

    Parameters:
    - documents (List[str]): List of documents to be added to the database.
    - name (str): Name of the collection.
    - embedding_function: Function used to generate embeddings for the documents.

    Returns:
    - chromadb.Collection: The created Chroma Collection instance.
    """
    chroma_client = chromadb.PersistentClient(path=path)
    database = chroma_client.create_collection(name=name, embedding_function=embedding_function)
    ids = [str(i) for i in range(len(documents))]
    database.add(ids=ids, documents=documents)

    return database


def load_chroma_database(name: str, embedding_function: SentenceTransformerEmbeddingFunction, path: str) -> chromadb.Collection: 
    """
    Loads an existing Chroma collection by name.

    Parameters:
    - name (str): Name of the collection to be loaded.
    - embedding_function: Embedding function for generating document embeddings.

    Returns:
    - chromadb.Collection: The loaded Chroma Collection instance.
    """
    chroma_client = chromadb.PersistentClient(path=path)
    database = chroma_client.get_collection(name=name, embedding_function=embedding_function)

    return database 


def project_embeddings(embeddings: List[np.ndarray], umap_transform:umap) -> np.ndarray:
    """
    Projects embeddings to a lower-dimensional space using UMAP.

    Parameters:
    - embeddings (List[np.ndarray]): List of embeddings to be projected.
    - umap_transform: A fitted UMAP model for dimensionality reduction.

    Returns:
    - np.ndarray: A 2D array of transformed embeddings.
    """
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])

    return umap_embeddings 


class SentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    # Since we do dynamic imports we have to type this as Any
    models: Dict[str, Any] = {}

    # If you have a beefier machine, try "gtr-t5-large".
    # for a full list of options: https://huggingface.co/sentence-transformers, https://www.sbert.net/docs/pretrained_models.html
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = False,
    ):
        if model_name not in self.models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ValueError(
                    "The sentence_transformers python package is not installed. Please install it with `pip install sentence_transformers`"
                )
            self.models[model_name] = SentenceTransformer(model_name, device=device)
        self._model = self.models[model_name]
        self._normalize_embeddings = normalize_embeddings

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(  # type: ignore
            list(input),
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
        ).tolist()