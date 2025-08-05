# backend/retrieval/vector_store.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import os

def load_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, filename))
            docs.extend(loader.load())
    return docs

def create_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def search(query, vector_store, k=3):
    results = vector_store.similarity_search(query, k=k)
    return [r.page_content for r in results]
