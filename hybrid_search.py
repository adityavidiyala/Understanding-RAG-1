from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from collections import defaultdict


load_dotenv()


# ──────────────────────────────────────────────────────────────────
# SETUP: Create our sample company data
# ──────────────────────────────────────────────────────────────────

chunks = [
    "Microsoft acquired GitHub for 7.5 billion dollars in 2018.",
    "Tesla Cybertruck production ramp begins in 2024.",
    "Google is a large technology company with global operations.",
    "Tesla reported strong quarterly results. Tesla continues to lead in electric vehicles. Tesla announced new manufacturing facilities.",
    "SpaceX develops Starship rockets for Mars missions.",
    "The tech giant acquired the code repository platform for software development.",
    "NVIDIA designs Starship architecture for their new GPUs.",
    "Tesla Tesla Tesla financial quarterly results improved significantly.",
    "Cybertruck reservations exceeded company expectations.",
    "Microsoft is a large technology company with global operations.", 
    "Apple announced new iPhone features for developers.",
    "The apple orchard harvest was excellent this year.",
    "Python programming language is widely used in AI.",
    "The python snake can grow up to 20 feet long.",
    "Java coffee beans are imported from Indonesia.", 
    "Java programming requires understanding of object-oriented concepts.",
    "Orange juice sales increased during winter months.",
    "Orange County reported new housing developments."
]

# Convert to Document objects for LangChain
documents = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]

print("Sample Data:")
for i, chunk in enumerate(chunks, 1):
    print(f"{i}. {chunk}")

print("\n" + "="*80)


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


vector_retriever = db.as_retriever(search_kwargs={"k": 2})

test_query = "space exploration company" #works in vector search but wouldn't work with keyword search

print(f"Testing: '{test_query}'")
test_docs = vector_retriever.invoke(test_query)
for doc in test_docs:
    print(f"Found: {doc.page_content}")



print("Setting up BM25 Retriever...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3



# Test exact keyword matching
# test_query = "space exploration company"
test_query = "Cybertruck"
# test_query = "Tesla"

print(f"Testing: '{test_query}'")
test_docs = bm25_retriever.invoke(test_query)
for doc in test_docs:
    print(f"Found: {doc.page_content}")



class HybridRetriever:
    def __init__(self, vector_retriever, bm25_retriever, w_vector=0.7, w_bm25=0.3, k=5):
        self.vector = vector_retriever
        self.bm25 = bm25_retriever
        self.w_vector = w_vector
        self.w_bm25 = w_bm25
        self.k = k

    def invoke(self, query):
        # Step 1: get docs
        docs_vector = self.vector.invoke(query)
        docs_bm25 = self.bm25.invoke(query)

        # Step 2: combine + deduplicate
        unique_docs = {}

        for doc in docs_vector + docs_bm25:
            key = doc.page_content.strip()
            if key not in unique_docs:
                unique_docs[key] = doc

        # Step 3: scoring (weighted rank)
        scores = defaultdict(float)

        for rank, doc in enumerate(docs_vector):
            scores[doc.page_content] += self.w_vector / (rank + 1)

        for rank, doc in enumerate(docs_bm25):
            scores[doc.page_content] += self.w_bm25 / (rank + 1)

        # Step 4: sort
        ranked_docs = sorted(
            unique_docs.values(),
            key=lambda d: scores[d.page_content],
            reverse=True
        )

        return ranked_docs[:self.k]



print("Setting up Hybrid Retriever...")

hybrid_retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever,
    w_vector=0.7,
    w_bm25=0.3,
    k=5
)

print("Setup complete!\n")


print("="*80)
print("HYBRID RETRIEVER TEST\n")

# Query 1: Mixed semantic + keyword
test_query = "purchase cost 7.5 billion"

print(f"Query: {test_query}\n")

retrieved_chunks = hybrid_retriever.invoke(test_query)

for i, doc in enumerate(retrieved_chunks, 1):
    print(f"{i}. {doc.page_content}")

print("\nExplanation:")
print("- Vector search understands 'purchase cost' → finds acquisition sentence")
print("- BM25 matches exact '7.5 billion'")
print("- Hybrid combines both → best result on top")