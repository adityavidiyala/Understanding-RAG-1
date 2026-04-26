from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  
from dotenv import load_dotenv


load_dotenv()

persistent_directory = "db/chroma_db"

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'}
)


db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)


query = "How much did Microsoft pay to acquire GitHub?"
print(f"Query: {query}\n")

#Method 1: Basic Similarity search *Returns top k most similar chunks*

# retriever = db.as_retriever(search_kwargs={"k": 3})
# relevant_docs = retriever.invoke(query)

# print(f"User Query: {query}")
# print("--- Context Found ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}: (Source: {doc.metadata.get('source', 'Unknown')})")

#     print(f"{doc.page_content}\n")
# print("-"*60)

#Method 2: Similarity search with a score threshold *Returns chunks above a certain similarity score*

# print("\n=== METHOD 2: Similarity with Score Threshold ===")
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 3,
#         "score_threshold": 0.3  # Only return docs with similarity >= 0.3
#     }
# )

# docs = retriever.invoke(query)
# print(f"Retrieved {len(docs)} documents (threshold: 0.3):\n")

# for i, doc in enumerate(docs, 1):
#     print(f"Document {i}:")
#     print(f"{doc.page_content}\n")

# print("-" * 60)


# ──────────────────────────────────────────────────────────────────
# METHOD 3: Maximum Marginal Relevance (MMR)
# Balances relevance and diversity - avoids redundant results
# ──────────────────────────────────────────────────────────────────

print("\n=== METHOD 3: Maximum Marginal Relevance (MMR) ===")
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,           # Final number of docs
        "fetch_k": 10,    # Initial pool to select from
        "lambda_mult": 0.5  # 0=max diversity, 1=max relevance
    }
)

docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents (λ=0.5):\n")

for i, doc in enumerate(docs, 1):
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")

print("=" * 60)
print("Done! Try different queries or parameters to see the differences.")