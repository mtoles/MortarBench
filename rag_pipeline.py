import os
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Constants
PDF_PATH = "Selling-Guide_12-10-25_Highlight.pdf"
FAISS_INDEX_PATH = "faiss_index_bge"

def load_and_split_documents(pdf_path: str):
    print(f"Loading {pdf_path}...")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    
    print("Splitting documents...")
    # Best parameters for chunking general technical documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")
    return splits

def build_retrievers(splits, force_rebuild=False):
    # Using one of the best open-source embedding models (MTEB leaderboard)
    # BAAI/bge-large-en-v1.5 provides excellent semantic search performance
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    
    if os.path.exists(FAISS_INDEX_PATH) and not force_rebuild:
        print("Loading existing FAISS indices...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Building FAISS index (this may take a couple of minutes locally)...")
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
            
    # FAISS Retriever setup with MMR for diverse retrieval
    faiss_retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={'k': 5, 'fetch_k': 20}
    )
    
    return faiss_retriever

def create_rag_chain(retriever):
    # Using Llama 3 via Ollama for a completely free, local LLM option
    # Note: You already have gemma3 installed, so this avoids your disk space limits
    llm = ChatOllama(model="gemma3", temperature=0)
    
    # Best prompt template for factual RAG
    template = """You are an expert mortgage assistant answering questions based on the provided Fannie Mae Selling Guide.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer or the context doesn't contain the answer, just say that you don't know. 
Do not make up information or base your answer on outside knowledge.
Provide a clear, detailed, and accurate answer based heavily on the policies mentioned. Quote specific sections if relevant.

Context:
{context}

Question:
{question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join([f"Content:\n{d.page_content}\nSource: {d.metadata.get('source', 'Unknown')} - Page: {d.metadata.get('page', 'Unknown')}" for d in docs])
        
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def main():
    parser = argparse.ArgumentParser(description="Run RAG on Selling Guide")
    parser.add_argument("--query", type=str, help="Question to ask the RAG system")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of index")
    args = parser.parse_args()
    
    if args.rebuild or not os.path.exists(FAISS_INDEX_PATH):
        splits = load_and_split_documents(PDF_PATH)
    else:
        splits = [] # Not needed if loading from disk
        
    retriever = build_retrievers(splits, force_rebuild=(args.rebuild or not os.path.exists(FAISS_INDEX_PATH)))
    rag_chain = create_rag_chain(retriever)
    
    if args.query:
        print(f"\nQuestion: {args.query}")
        result = rag_chain.invoke(args.query)
        print(f"\nAnswer:\n{result}")
    else:
        print("\nInteractive RAG session started. Type 'exit' to quit.")
        while True:
            try:
                query = input("\nEnter your question: ")
                if query.lower() in ['exit', 'quit']:
                    break
                if not query.strip():
                    continue
                result = rag_chain.invoke(query)
                print(f"\nAnswer:\n{result}")
            except EOFError:
                break
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
