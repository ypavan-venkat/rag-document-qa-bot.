import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

SYSTEM_PROMPT = (
    "You are an intelligent assistant for document-based question-answering. "
    "Use the following pieces of retrieved context to answer the question. "
    "If the answer is not contained within the context, you must strictly reply with: "
    "'I cannot answer this based on the provided documents.' "
    "Do not use your external training data to answer.\n\n"
    "Context:\n{context}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    # 1. Initialize Local Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists("./chroma_db"):
        print("Vector DB not found. Please run ingest.py first.")
        return
        
    # Using k=10 to give the bot a massive context window!
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 2. Initialize the MODERN Gemini LLM (2.5-flash)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # 3. Create Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    # 4. Build the RAG Chain using modern LCEL syntax
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    print("\n" + "="*50)
    print("🤖 RAG Document Q&A Bot Initialized")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    # 5. Interactive Command Line Loop
    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down bot. Goodbye!")
            break
        if not user_input.strip():
            continue

        response = rag_chain_with_source.invoke(user_input)
        
        print(f"\n🤖 Bot: {response['answer']}")
        print("\n📚 Sources Cited:")
        
        unique_sources = set()
        for doc in response['context']:
            source = doc.metadata.get('source', 'Unknown File')
            page = doc.metadata.get('page', 'Unknown Page')
            file_name = os.path.basename(source)
            unique_sources.add(f"- File: {file_name} | Page: {page}")
            
        for citation in unique_sources:
            print(citation)

if __name__ == "__main__":
    main()