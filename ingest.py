import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def build_vector_db():
    print("Starting document ingestion...")
    loader = PyPDFDirectoryLoader("./data")
    docs = loader.load()
    
    if not docs:
        print("No documents found in the /data directory.")
        return

    print(f"Loaded {len(docs)} document pages.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split documents into {len(chunks)} chunks.")

    # Using local open-source embeddings (No API key needed for this part!)
    print("Downloading local embedding model (this takes a minute the first time)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Embedding chunks and saving to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Success! Vector database created and persisted to disk.")

if __name__ == "__main__":
    build_vector_db()