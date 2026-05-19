import os
import argparse
import threading
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from llm import call_llm_wrapper
from sentence_transformers import CrossEncoder

load_dotenv()

# Constants
PDF_PATH = "Selling-Guide_12-10-25_Highlight.pdf"
FAISS_INDEX_PATH = "faiss_index_bge_semantic"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

class LocalCrossEncoderReranker:
    """A simple reranker using sentence-transformers CrossEncoder."""
    def __init__(self, model_name=RERANKER_MODEL_NAME, top_n=10):
        if CrossEncoder is None:
            raise ImportError("sentence-transformers is not installed.")
        print(f"Loading CrossEncoder model {model_name}...")
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        if not documents:
            return []
        
        # Deduplicate docs based on content
        unique_docs = []
        seen_content = set()
        for doc in documents:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
                
        pairs = [[query, doc.page_content] for doc in unique_docs]
        scores = self.model.predict(pairs)
        
        scored_docs = list(zip(unique_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        top_docs = []
        for doc, score in scored_docs[:self.top_n]:
            # Create a new document to avoid modifying the original cached one
            new_doc = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
            new_doc.metadata["relevance_score"] = float(score)
            top_docs.append(new_doc)
            
        return top_docs


_SHARED_RERANKER = None
_RERANKER_LOCK = threading.Lock()

def get_shared_reranker():
    global _SHARED_RERANKER
    if _SHARED_RERANKER is None:
        with _RERANKER_LOCK:
            if _SHARED_RERANKER is None:
                print("Initializing CrossEncoder for reranking...")
                _SHARED_RERANKER = LocalCrossEncoderReranker(model_name=RERANKER_MODEL_NAME, top_n=10)
    return _SHARED_RERANKER

def retrieve_and_rerank(model_id, query: str, retriever) -> List[Document]:
    # 1. Custom Query Rewriting
    print(f"\n[RAG] Logic triggered using model: {model_id}")
    print(f"\n[RAG] Generating multi-queries...")
    sub_queries = generate_queries(model_id, query)
    print(f"[RAG] Generated sub-queries: {sub_queries}")
    
    all_docs = []
    for sq in sub_queries:
        try:
            all_docs.extend(retriever.invoke(sq))
        except Exception as e:
            print(f"Retrieval failed for '{sq}': {e}")
            
    print(f"[RAG] Retrieved {len(all_docs)} raw documents across all sub-queries.")
        
    # 2. CrossEncoder Reranking
    reranker = get_shared_reranker()
    print(f"[RAG] Reranking {len(all_docs)} documents with CrossEncoder...")
    reranked_docs = reranker.compress_documents(all_docs, query)
    print(f"[RAG] Retained top {len(reranked_docs)} documents.")
    
    return reranked_docs

def load_and_split_documents(pdf_path: str):
    print(f"Loading {pdf_path}...")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    
    print("Splitting documents using SemanticChunker (this takes longer but preserves meaning)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Semantic chunking relies on embedding similarity to determine breaks
    text_splitter = SemanticChunker(
        embeddings, breakpoint_threshold_type="percentile"
    )
    
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} semantic chunks.")
    return splits

def build_retrievers(model_id, splits, force_rebuild=False):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    if os.path.exists(FAISS_INDEX_PATH) and not force_rebuild:
        print("Loading existing FAISS indices...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Building FAISS index (this may take a couple of minutes locally)...")
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
            
    # Significantly increase k value to ensure we catch facts hidden deep in the vector space
    # The CrossEncoder will filter this down later.
    faiss_retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={'k': 50, 'fetch_k': 100}
    )
    
    return faiss_retriever

def generate_queries(model_id, query: str, num_queries: int = 3) -> List[str]:
    prompt = f"""You are an expert mortgage RAG assistant. Your task is to rewrite the user's query into {num_queries} specific, fact-seeking queries to retrieve relevant policies from a Fannie Mae Selling Guide vector database.
Instead of just rephrasing the question, break it down into the core definitions and policies needed to answer it. 
For example, if the user's question is "which txns are large deposits", you should produce a RAG query like "What is the definition of a large deposit".
Target specific facts, thresholds, and definitions that would be documented in the guidelines. Provide these {num_queries} queries separated by newlines, with no other text.
Original question: {query}"""
    try:
        resp, _, _ = call_llm_wrapper(model_id, [{"role": "user", "content": prompt}])
        queries = [q.strip("- \t*1234567890.") for q in resp.strip().split("\n") if q.strip()]
        if not queries:
            return [query]
        # Return unique queries including original
        all_queries = list(set([query] + queries))
        return all_queries
    except Exception as e:
        print(f"Query generation failed: {e}")
        return [query]

def create_rag_chain(model_id, retriever):
    print(f"[RAG] Initializing LangChain for model: {model_id}")
    llm = ChatOllama(model=model_id, temperature=0)
    
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
    
    def retrieve_and_format(query: str) -> str:
        reranked_docs = retrieve_and_rerank(model_id, query, retriever)
        formatted = "\n\n".join([f"Content:\n{d.page_content}\nSource: {d.metadata.get('source', 'Unknown')} - Page: {d.metadata.get('page', 'Unknown')} - Score: {d.metadata.get('relevance_score', 0):.4f}" for d in reranked_docs])
        return formatted
        
    chain = (
        {"context": retrieve_and_format, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def main():
    parser = argparse.ArgumentParser(description="Run RAG on Selling Guide")
    parser.add_argument("--query", type=str, help="Question to ask the RAG system")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of index")
    parser.add_argument("--model", type=str, default="gemma3")
    args = parser.parse_args()
    
    if args.rebuild or not os.path.exists(FAISS_INDEX_PATH):
        splits = load_and_split_documents(PDF_PATH)
    else:
        splits = [] # Not needed if loading from disk
        
    retriever = build_retrievers(args.model, splits, force_rebuild=(args.rebuild or not os.path.exists(FAISS_INDEX_PATH)))
    rag_chain = create_rag_chain(args.model, retriever)
    
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
