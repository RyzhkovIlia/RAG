from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections
)
from src.tools import (
    load_pdf_file,
    split_text_to_token,
    SentenceTransformerEmbeddings,
    save_params_for_retriever
)
from langchain_milvus.utils.sparse import BM25SparseEmbedding

CONNECTION_URI = "http://localhost:19530"
connections.connect(uri=CONNECTION_URI)

# Название новой коллекции
name_of_collection = 'Pushkin'

# Создаем схему новой коллекции
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

# Инициализируем новую коллекцию 
schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
collection = Collection(
    name=name_of_collection, schema=schema, consistency_level="Strong"
)

# Добавляем индексы в новую коллекцию
dense_index = {"index_type": "FLAT", "metric_type": "IP"}
collection.create_index("dense_vector", dense_index)
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
collection.create_index("sparse_vector", sparse_index)
collection.flush()

# Получаем текст из PDF файла
pdf_text = load_pdf_file(file_path='dataset/skazka_o_rubake_i_rubke.pdf')
chunked_text = split_text_to_token(text=pdf_text)

# Инициализируем два типа Эмбеддинга
dense_embedding_func = SentenceTransformerEmbeddings()
sparse_embedding_func = BM25SparseEmbedding(corpus=chunked_text)

# Добавляем текст в коллекцию и сохраняем
entities = []
for text in chunked_text:
    entity = {
        dense_field: dense_embedding_func.embed_documents([text])[0],
        sparse_field: sparse_embedding_func.embed_documents([text])[0],
        text_field: text,
    }
    entities.append(entity)
collection.insert(entities)
collection.load()

# Сохраняем все параметры коллекции
save_params_for_retriever(
    collection=collection, 
    dense_field=dense_field, 
    sparse_field=sparse_field,
    dense_embedding_func=dense_embedding_func,
    sparse_embedding_func=sparse_embedding_func
)