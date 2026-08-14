from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def setup_vector_db(pdf_path):
    """Setup vector database from PDF"""
    # Load and chunk PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    # Create vector database
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db
