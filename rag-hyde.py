import os
import json
from collections import defaultdict
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def load_and_split_documents(pdf_path):
    """Load PDF and split into chunks"""
    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    split_docs = text_splitter.split_documents(docs)
    
    print("Number of documents before splitting:", len(docs))
    print("Number of documents after splitting:", len(split_docs))
    
    return split_docs

def setup_vector_store(split_docs, embedder):
    """Initialize vector store with documents"""
    vector_store = QdrantVectorStore.from_documents(
        documents=split_docs,
        url="http://localhost:6333",
        collection_name="learning_langchain",
        embedding=embedder
    )
    return vector_store

def generate_document(client, user_query):
    """Break out the user query into multiple smaller steps"""
    GENERATE_DOCUMENT_SYSTEM_PROMPT = """
    You are a helpful assistant. You will be provided with a question and you need to write a proper document on the topics included in it. Use proper technical phrases and terms used in the related industry. 
    """

    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[
            {"role": "system", "content": GENERATE_DOCUMENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_query
            }
        ]
    )
    content = response.choices[0].message.content
    print("Generate Document response:", content)
    
    return content

def similarity_search(vector_store, query):
    """Perform similarity search for a given query"""
    relevant_chunks = vector_store.similarity_search(query=query)
    return relevant_chunks

def retrieval_generation(client, query, context_docs):
    """Generate an answer based on query and context"""
    # Format context from documents
    context = "\n\n".join([doc.page_content for doc in context_docs])
    print(context)
    
    GENERATION_SYSTEM_PROMPT = f"""
    You are a helpful assistant. You will be provided with a question and relevant context filtered according to user's query. 
    Your task is to provide a concise answer based on the context.
    
    Context: {context}
    """

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

def main():
    # Initialize components
    pdf_path = Path("./nodejs.pdf")
    split_docs = load_and_split_documents(pdf_path)
    
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
    )
    
    vector_store = setup_vector_store(split_docs, embedder)
    
    # Create client for chatting
    client = OpenAI(
        api_key=GOOGLE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    # Main interaction loop
    while True:
        user_query = input(">> ")
        if user_query.lower() in ["exit", "quit", "q"]:
            break
            
        # Generate related questions
        content = generate_document(client, user_query)
        
        # Final generation that uses all previous context
        relevant_chunks = similarity_search(vector_store, content)
        print(f"Final query: {len(relevant_chunks)} relevant chunks found.")
        final_generation = retrieval_generation(client, content, relevant_chunks)
        print(f"Final Answer: {final_generation}")
        
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()