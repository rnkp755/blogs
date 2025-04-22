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

def generate_questions(client, user_query):
    """Fan out the user query into multiple related questions"""
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
    print("Generated questions:", questions)
    
    # Include the original query as well
    all_questions = [user_query] + questions
    return all_questions

def similarity_search(vector_store, query):
    """Perform similarity search for a given query"""
    relevant_chunks = vector_store.similarity_search(query=query)
    return relevant_chunks

def reciprocal_rank_fusion(all_results, k=60, threshold=0.0):
    """Combine multiple search results using Reciprocal Rank Fusion"""
    # Dictionary to store document content by ID
    doc_content = {}
    
    # Calculate RRF scores
    rrf_scores = defaultdict(float)
    
    # Iterate over each ranking list
    for query_results in all_results:
        for rank, document in enumerate(query_results):
            doc_id = document.metadata["_id"]
            doc_content[doc_id] = document  # Store the actual document
            rrf_scores[doc_id] += 1 / (k + rank + 1)
    
    # Filter by threshold and sort by score
    filtered = [(doc_id, score) for doc_id, score in rrf_scores.items() if score >= threshold]
    sorted_results = sorted(filtered, key=lambda x: x[1], reverse=True)
    
    # Return the actual documents, not just IDs and scores
    return [doc_content[doc_id] for doc_id, _ in sorted_results]

def retrieval_generation(client, query, context_docs):
    """Generate an answer based on query and context"""
    # Format context from documents
    context = "\n\n".join([doc.page_content for doc in context_docs])
    print(context)
    
    GENERATION_SYSTEM_PROMPT = f"""
    You are a helpful assistant. You will be provided with a question and relevant context from a document. 
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
        all_questions = generate_questions(client, user_query)
        
        # Collect search results for all questions
        all_results = []
        for question in all_questions:
            results = similarity_search(vector_store, question)
            all_results.append(results)
        
        # Combine results using RRF
        fused_results = reciprocal_rank_fusion(all_results)
        print(f"Found {len(fused_results)} relevant documents after fusion")
        
        # Generate response
        final_answer = retrieval_generation(client, user_query, fused_results)
        print("\nAnswer:")
        print(final_answer)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
