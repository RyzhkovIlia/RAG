import os
from src.tools import load_chroma_database, SentenceTransformerEmbeddingFunction
from src.promt import request_db
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--request", required=True, type=str, help="Your request to database")
parser.add_argument("--api", required=True, type=str, help="Your Google Gemini API")
parser.add_argument("--name_db", required=True, type=str, help="Name of vectorize database")
parser.add_argument("--path", required=True, type=str, help="Path to your database")
args = parser.parse_args()

if args.api == 'exist':
    API_KEY = os.getenv("GEMINI_API_KEY")
else:
    os.environ["GEMINI_API_KEY"]=args.api
    API_KEY = os.getenv("GEMINI_API_KEY")

print(API_KEY)
database = load_chroma_database(name=args.name_db, embedding_function=SentenceTransformerEmbeddingFunction(), path=args.path)

answer = request_db(database=database, query=args.request, api_gemino=API_KEY)

print(answer)