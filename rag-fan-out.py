"""
- pip install langchain_community pypdf langchain_text_splitters langchain-google-genai langchain_qdrant openai
"""

import os
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

pdf_path = Path("./nodejs.pdf")

loader = PyPDFLoader(file_path = pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(docs)

print("Number of documents before splitting:", len(docs))
# print(docs[0])  # docs is a list of Document objects
print("Number of documents after splitting:", len(split_docs))
# print(split_docs[0])    # split_docs is a list of Document objects

embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
    )


vector_store = QdrantVectorStore.from_documents(
    documents = split_docs,
    url = "http://localhost:6333",
    collection_name = "learning_langchain",
    embedding = embedder
)


retriever = QdrantVectorStore.from_existing_collection(
    url = "http://localhost:6333",
    collection_name = "learning_langchain",
    embedding = embedder
)

# Create client for chatting
client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

finalResponse = ""

def fan_out():
    user_query = input(">> ")

    global finalResponse
    finalResponse = ""  # Reset the final response for each new query

    FAN_OUT_SYSTEM_PROMPT = """
    You are a helpful assistant. You will be provided with a question and you need to generate 3 questions out of it focusing on different aspects of it or related to it. The focus should be on what user might be interested in and maybe he couldn't ask it directly. You need to generate them.

    Rules:
    - Follow the output JSON format.

    Example:
    User Query: How does garbage collection workin python?
    Output: {{ "q1": "What triggers garbage collection in python?", "q2": "Garbage collection algorithms in Python?", "q3": "How memory leaks relate to GC?" }}
    """

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": FAN_OUT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_query
            }
        ]
    )
    content = response.choices[0].message.content
    print("Fan out response:", content)
    # Parse the JSON response
    parsed_response = json.loads(content)
    # Extract the questions
    questions = [parsed_response["q1"], parsed_response["q2"], parsed_response["q3"]]
    print("Questions:", questions)
    # Call the retrieval_generation function for each question
    for question in questions:
        retrieval_generation(question)

    print("Final response:", finalResponse)
    
def retrieval_generation(query: str):

    global finalResponse

    relevant_chunks  = retriever.similarity_search(
        query = query
    )

    print("Search result:", relevant_chunks)

    GENERATION_SYSTEM_PROMPT = """
    You are a helpful assistant. You will be provided with a question and relevant context from a document. Your task is to provide a concise answer based on the context.
    Context: {relevant_chunks}
    """

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT.format(relevant_chunks=relevant_chunks)},
            {
                "role": "user",
                "content": query
            }
        ]
    )
    answer = response.choices[0].message.content
    finalResponse += f"\n{answer}"


fan_out()
