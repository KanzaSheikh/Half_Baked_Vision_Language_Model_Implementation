from llm import llm
from web_scraping_agent import get_web_content


def get_local_content(vector_db, query):
    """Get content from vector database"""
    docs = vector_db.similarity_search(query, k=5)
    print("Documents (local content): ", docs)
    return " ".join([doc.page_content for doc in docs])


def check_local_knowledge(query, context):
    """Router function to determine if we can answer from local knowledge"""
    prompt = '''Role: Question-Answering Assistant
Task: Determine whether the system can answer the user's question based on the provided text.
Instructions:
    - Analyze the text and identify if it contains the necessary information to answer the user's question.
    - Provide a clear and concise response indicating whether the system can answer the question or not.
    - Your response should include only a single word. Nothing else, no other text, information, header/footer.
Output Format:
    - Answer: Yes/No
Study the below examples and based on that, respond to the last question.
Examples:
    Input:
        Text: The capital of France is Paris.
        User Question: What is the capital of France?
    Expected Output:
        Answer: Yes
    Input:
        Text: The population of the United States is over 330 million.
        User Question: What is the population of China?
    Expected Output:
        Answer: No
    Input:
        User Question: {query}
        Text: {text}
'''
    formatted_prompt = prompt.format(text=context, query=query)
    response = llm.invoke(formatted_prompt)
    return response.content.strip().lower() == "yes"


def generate_final_answer(context, query):
    """Generate final answer using LLM"""
    messages = [
        (
            "system",
            "You are a helpful assistant. Use the provided context to answer the query accurately.",
        ),
        ("system", f"Context: {context}"),
        ("human", query),
    ]
    response = llm.invoke(messages)
    return response.content


def process_query(query, vector_db, local_context):
    """Main function to process user query"""
    print(f"Processing query: {query}")

    # Step 1: Check if we can answer from local knowledge
    can_answer_locally = check_local_knowledge(query, local_context)
    print(f"Can answer locally: {can_answer_locally}")

    # Step 2: Get context either from local DB or web
    if can_answer_locally:
        context = get_local_content(vector_db, query)
        print("Retrieved context from local documents")
    else:
        context = get_web_content(query)
        print("Retrieved context from web scraping")

    # Step 3: Generate final answer
    answer = generate_final_answer(context, query)
    return answer
