# ingest.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader

VECTOR_DIR = "vectordb/index"

def load_and_index_documents(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF or TXT.")

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    vectordb.save_local(VECTOR_DIR)
    return True
