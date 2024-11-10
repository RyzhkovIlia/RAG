import re

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections
)

from langchain_milvus.utils.sparse import BM25SparseEmbedding

from src.tools import (
    load_pdf_file,
    split_text_to_token,
    SentenceTransformerEmbeddings,
    save_params_for_retriever
)

# connect to Milvus
CONNECTION_URI = "http://localhost:19530"
connections.connect(uri=CONNECTION_URI)

# Name of the new collection
name_of_collection = 'Tresis'

# Creating a scheme for a new collection
pk_field = "doc_id"
dense_field = "dense_vector"
sparse_field = "sparse_vector"
text_field = "text"
fields = [
    FieldSchema(
        name=pk_field,
        dtype=DataType.VARCHAR,
        is_primary=True,
        auto_id=True,
        max_length=100,
    ),
    FieldSchema(name=dense_field, dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name=sparse_field, dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name=text_field, dtype=DataType.VARCHAR, max_length=65_535),
]

# Initialize a new collection  
schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
collection = Collection(
    name=name_of_collection, schema=schema, consistency_level="Strong"
)

# Add indexes to a new collection
dense_index = {"index_type": "FLAT", "metric_type": "IP"}
collection.create_index("dense_vector", dense_index)
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
collection.create_index("sparse_vector", sparse_index)
collection.flush()

# Getting text from PDF file
pdf_text = load_pdf_file(file_path='dataset/example.pdf', start_page=13, end_page=158)

# Text processing
pdf_text = [p.replace('\n', ' ').strip() for p in pdf_text if (not re.search(r'© Springer International Publishing Switzerland 2014', p.replace('\n', ' ').strip())) and (not p.replace('\n', ' ').strip().isdigit())]
pdf_text_2 = []
for i in pdf_text:
    if i.split(' ')[0].istitle() or i.split(' ')[0] == '•':
        pdf_text_2.append(i)
    else:
        pdf_text_2[-1]+=i
chunked_text = split_text_to_token(text_split=pdf_text_2)
chunked_text_2 = []
for i in chunked_text:
    if not len(i.strip())<5:
        chunked_text_2.append(i.strip())

# Initialize two types of Embedding
dense_embedding_func = SentenceTransformerEmbeddings()
sparse_embedding_func = BM25SparseEmbedding(corpus=chunked_text)

# Add the text to the collection and save it
entities = []
for text in chunked_text_2:
    entity = {
        dense_field: dense_embedding_func.embed_documents([text])[0],
        sparse_field: sparse_embedding_func.embed_documents([text])[0],
        text_field: text,
    }
    entities.append(entity)
collection.insert(entities)
collection.load()

# Save all collection settings
save_params_for_retriever(
    collection=collection, 
    dense_field=dense_field, 
    sparse_field=sparse_field,
    dense_embedding_func=dense_embedding_func,
    sparse_embedding_func=sparse_embedding_func
)