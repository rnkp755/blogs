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

def generate_steps(client, user_query):
    """Break out the user query into multiple smaller steps"""
    FAN_OUT_SYSTEM_PROMPT = """
    You are a helpful assistant. You will be provided with a question and you need to break it into 3 simpler & sequential steps to solve the problem. What steps do you think would be best to solve the problem?

    Rules:
    - Follow the output JSON format.
    - The `content` in output JSON must be a list of steps.

    Example:
    User Query: How to handle file-uploads on server?
    Output: { "type": "steps", "content": ["Accept file from req.files. Take help of multer to do that.", "Upload file to the S3 bucket or any other db and take out public url", "Store that public url in actual database"] }
    """

    response = client.chat.completions.create(
        model="gemini-1.5-flash",
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
    print("Query Breaker response:", content)
    
    # Parse the JSON response
    parsed_response = json.loads(content)
    
    # Extract the steps
    steps = parsed_response["content"]
    print("Generated steps:", steps)
    
    return steps

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
    You are a helpful assistant. You will be provided with a question and relevant context filtered from sub-steps of the same query. 
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
        steps = generate_steps(client, user_query)
        
        # Process each step and build on previous answers
        previous_generation = ""
        for i, step in enumerate(steps):
            # If we have previous generation, add it to the step query
            if previous_generation:
                enhanced_step = f"{step} Previous Generation: {previous_generation}"
            else:
                enhanced_step = step
                
            # Retrieve relevant chunks for this step
            relevant_chunks = similarity_search(vector_store, enhanced_step)
            print(f"Step {i+1}/{len(steps)}: {len(relevant_chunks)} relevant chunks found.")
            
            # Generate answer for this step
            generation = retrieval_generation(client, enhanced_step, relevant_chunks)
            print(f"Step {i+1} Generation: {generation}")
            
            # Store this generation for the next step
            previous_generation = generation
        
        # Final generation that uses all previous context
        final_query = user_query + f" Previous Generation: {previous_generation}"
        relevant_chunks = similarity_search(vector_store, final_query)
        print(f"Final query: {len(relevant_chunks)} relevant chunks found.")
        final_generation = retrieval_generation(client, final_query, relevant_chunks)
        print(f"Final Answer: {final_generation}")
        
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()