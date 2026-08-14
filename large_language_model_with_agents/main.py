import os
from dotenv import load_dotenv

from vector_db import setup_vector_db
from answer import get_local_content
from answer import process_query


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI=os.getenv("GEMINI")
os.environ["HF_TOKEN"] = "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"


def main():
    # Setup
    print("in the main method")
    pdf_path = "genai-principles.pdf"

    print("PDF path: ", pdf_path)

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Initialize vector database
    print("Setting up vector database...")
    vector_db = setup_vector_db(pdf_path)

    # Get initial context from PDF for routing
    local_context = get_local_content(vector_db, "")

    # Example usage
    query = ("What is Agentic RAG?")
    result = process_query(query, vector_db, local_context)
    print("\nFinal Answer:")
    print(result)
#

if __name__ == "__main__":
    main()
