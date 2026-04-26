from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from collections import defaultdict
from langchain_cohere import CohereRerank
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# pip install langchain_cohere

load_dotenv()


chunks = [
    # Tesla - Financial & Production
    "Tesla reported record quarterly revenue of $25.2 billion in Q3 2024.",
    "Tesla's automotive gross margin improved to 19.3% this quarter.",
    "Tesla Cybertruck production ramp begins in 2024 with initial deliveries.",
    "Tesla announced plans to expand Gigafactory production capacity.",
    "Tesla stock price reached new highs following earnings announcement.",
    "Tesla's energy storage business grew 40% year-over-year.",
    "Tesla continues to lead in electric vehicle market share globally.",
    "Tesla Model Y became the best-selling vehicle worldwide.",
    "Tesla reported strong free cash flow generation of $7.5 billion.",
    "Tesla's Full Self-Driving revenue increased significantly.",
    
    # Microsoft - Development & Acquisitions
    "Microsoft acquired GitHub for $7.5 billion in 2018.",
    "Microsoft's cloud revenue Azure grew 29% year-over-year.",
    "Microsoft announced new AI features for Visual Studio Code.",
    "Microsoft Teams integration with GitHub enhances developer workflow.",
    "Microsoft's developer tools division sees strong adoption.",
    "Microsoft acquired Activision Blizzard for $68.7 billion.",
    "Microsoft's productivity suite gained 50 million new users.",
    "Microsoft announced new Surface devices for developers.",
    "Microsoft's AI Copilot features expand to more development tools.",
    "Microsoft's enterprise solutions drive revenue growth.",
    
    # NVIDIA - AI & Hardware
    "NVIDIA's data center revenue reached $47.5 billion annually.",
    "NVIDIA's H100 GPUs see unprecedented demand for AI training.",
    "NVIDIA announced next-generation Blackwell architecture.",
    "NVIDIA's gaming revenue declined due to crypto market changes.",
    "NVIDIA's automotive AI platform partnerships expanded.",
    "NVIDIA's AI chip shortage affects cloud providers.",
    "NVIDIA stock valuation exceeds $2 trillion market cap.",
    "NVIDIA's CUDA platform dominates AI development.",
    "NVIDIA announced new AI inference chips for edge computing.",
    "NVIDIA's partnership with major cloud providers strengthens.",
    
    # Google/Alphabet - AI & Cloud
    "Google's AI investments total over $100 billion in recent years.",
    "Google Cloud revenue grew 35% reaching $8.4 billion quarterly.",
    "Google announced Gemini AI model competing with GPT-4.",
    "Google's search advertising revenue remains strong at $59 billion.",
    "Google's Workspace products integrate advanced AI features.",
    "Google announced quantum computing breakthroughs.",
    "Google's autonomous vehicle division Waymo expands operations.",
    "Google's AI research published breakthrough papers.",
    "Google's cloud AI services see enterprise adoption.",
    "Google faces regulatory scrutiny over AI dominance.",
    
    # Noisy/Less Relevant Chunks
    "The Tesla coil was invented by Nikola Tesla in 1891.",
    "Microsoft Excel spreadsheet formulas can be complex for beginners.",
    "NVIDIA Shield TV streaming device gets software update.",
    "Google Maps navigation improved with real-time traffic data.",
    "Production delays affected multiple manufacturing sectors.",
    "Financial markets showed volatility during earnings season.",
    "Revenue recognition standards changed for software companies.",
    "Hardware components face supply chain constraints globally.",
    "Development tools market grows with remote work trends.",
    "AI research requires significant computational resources.",
    "Quarterly reports show mixed results across tech sector.",
    "Stock market analysts upgrade technology sector ratings.",
    "Cloud computing adoption accelerates in enterprise market.",
    "Data center construction increases globally.",
    "Semiconductor shortage impacts various industries.",
    "Electric vehicle charging infrastructure expands rapidly.",
    "Software development productivity tools gain popularity.",
    "Machine learning frameworks become more accessible.",
    "Enterprise software licensing models evolve.",
    "Technology conferences showcase latest innovations."
]

print(f"Created {len(chunks)} sample chunks for demonstration")



# Convert to Document objects
documents = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]



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


vector_retriever = db.as_retriever(search_kwargs={"k": 15})

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 15


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

query = "Tesla financial performance and production updates"

print("STEP 1: Hybrid Search Results")
print("-"*50)


retrieved_docs = hybrid_retriever.invoke(query)

for i, doc in enumerate(retrieved_docs, 1):
    print(f"{i:2d}. {doc.page_content}")

print(f"\n(Retrieved {len(retrieved_docs)} total chunks for reranking)\n")

print("-"*50)
print()

print("STEP 2: After Cohere Reranking (Top 10)")
print("-"*50)

# Initialize Cohere reranker
reranker = CohereRerank(model="rerank-english-v3.0", top_n=10)

# Rerank the retrieved documents
reranked_docs = reranker.compress_documents(retrieved_docs, query)

# Show reranked results
for i, doc in enumerate(reranked_docs, 1):
    print(f"{i:2d}. {doc.page_content}")

print("\n" + "="*80)
print("ANALYSIS:")
print("✅ Hybrid Search: Mixed relevant and irrelevant results")
print("✅ Reranking: Most relevant Tesla financial/production info at top")
print("✅ Notice how reranking moved the most contextually relevant chunks higher")

# Optional: Show the difference more clearly
print("\n" + "="*80)
print("KEY IMPROVEMENTS AFTER RERANKING:")
print("-"*40)


hybrid_top_5 = [doc.page_content for doc in retrieved_docs[:5]]
reranked_top_5= [doc.page_content for doc in reranked_docs[:5]]

print("BEFORE (Hybrid Top 3):")
for i, content in enumerate(hybrid_top_5, 1):
    print(f"  {i}. {content}")

print("\nAFTER (Reranked Top 3):")
for i, content in enumerate(reranked_top_5, 1):
    print(f"  {i}. {content}")


print("\n" + "="*80)
print("FINAL: RAG with Reranked Context")
print("-"*40)

# Use top 5 reranked documents for final answer
top_reranked = reranked_docs[:5]

combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in top_reranked])}
Please provide a clear, helpful answer using only the information from these documents."""

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