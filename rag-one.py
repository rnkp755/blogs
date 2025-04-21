import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

pdf_path = Path("./gate-pyqs.pdf")

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

user_query = input(">> ")

relevant_chunks  = retriever.similarity_search(
    query = user_query
)

print("Search result:", relevant_chunks)

SYSTEM_PROMPT = """
You are a helpful assistant. You will be provided with a question and relevant context from a document. Your task is to provide a concise answer based on the context.
Context: {relevant_chunks}
"""

# Create client for chatting
client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT.format(relevant_chunks=relevant_chunks)},
        {
            "role": "user",
            "content": user_query
        }
    ]
)

print(response.choices[0].message)
