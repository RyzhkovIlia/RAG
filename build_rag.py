import os
from src.tools import load_pdf_file, split_text_to_token, create_chroma_database, SentenceTransformerEmbeddingFunction
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--pdf", required=True, type=str, help="Path to pdf file")
parser.add_argument("--name_db", required=True, type=str, help="Name of vectorize database")
parser.add_argument("--path", required=True, type=str, help="Path to your database")
args = parser.parse_args()

pdf_text = load_pdf_file(file_path=args.pdf)
chunked_text = split_text_to_token(text=pdf_text)
db = create_chroma_database(documents=chunked_text, path=args.path, name=args.name_db, embedding_function=SentenceTransformerEmbeddingFunction())
print(f'New database created successful! Name database is {args.name_db}, path to database - {args.path}.')