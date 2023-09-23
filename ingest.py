import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration dictionary for various parameters
CONFIG = {
    "DATA_PATH": os.getenv("DATA_PATH"),
    "DB_FAISS_PATH": os.getenv("DB_FAISS_PATH"),
    "MODEL_NAME": 'sentence-transformers/all-MiniLM-L6-v2',
    "DEVICE": 'cuda',
    "CHUNK_SIZE": 500,
    "CHUNK_OVERLAP": 50
}

def load_txt_files(data_path):
    txt_files = list(Path(data_path).glob("*.txt"))
    txt_documents = []

    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            txt_documents.append(f.read())

    return txt_documents

def create_vector_db():
    pdf_loader = DirectoryLoader(CONFIG["DATA_PATH"], glob='*.pdf', loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()
    
    excel_loader = DirectoryLoader(CONFIG["DATA_PATH"], glob='*.xlsx', loader_cls=UnstructuredExcelLoader)
    excel_documents = [doc.load(mode="elements") for doc in excel_loader.load()]

    txt_documents = load_txt_files(CONFIG["DATA_PATH"])

    documents = pdf_documents + excel_documents + txt_documents

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CONFIG["CHUNK_SIZE"], chunk_overlap=CONFIG["CHUNK_OVERLAP"])
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=CONFIG["MODEL_NAME"], model_kwargs={'device': CONFIG["DEVICE"]})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(CONFIG["DB_FAISS_PATH"])

if __name__ == "__main__":
    create_vector_db()