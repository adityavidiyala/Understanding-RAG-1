import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


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

model = ChatGroq(model="llama-3.1-8b-instant")

history = []

def ask_question(question):
    print(f"\n--- You asked: {question} ---")

    if history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + history + [HumanMessage(content=f"New question: {question}")]

        res = model.invoke(messages)
        search_question = res.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = question


    retriever = db.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")

    
    docs_text = "\n\n".join(
        [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    )

    combined_input = f"""Based on the following documents, answer the question.

        Question: {question}

        Documents:
        {docs_text}

        Rules:
        - Use ONLY the information from the documents
        - If not found, say: "I don't have enough information based on the provided documents."
    """

       
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    # Step 5: Remember this conversation
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer

def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()
    




