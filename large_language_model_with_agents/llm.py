from langchain_groq import ChatGroq


llm = ChatGroq(
    # model="llama-3.3-70b-specdec",
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
)
