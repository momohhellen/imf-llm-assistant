# qa_engine.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

VECTOR_DIR = "vectordb/index"

def query_vector_store(query: str):
    vectordb = FAISS.load_local(VECTOR_DIR, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.run(query)
    return result
