import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

# 1. Load the EXACT SAME embedding model used during ingestion
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'}
)

# 2. Load the existing vector store
# Note: Ensure the folder 'db/chroma_db' exists before running this
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# 3. Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# k=5 means it will return the top 5 most similar chunks
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3 
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# print("--- Context Found ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}: (Source: {doc.metadata.get('source', 'Unknown')})")
#     print(f"{doc.page_content}\n")



# Create the combined input (Context + Query)
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# Initialize Groq model
# The "versatile" models are great for RAG tasks
model = ChatGroq(model="llama-3.1-8b-instant")

# Define the messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model
result = model.invoke(messages)

print("\n--- Generated Response---")
print("Content only:")
print(result.content)