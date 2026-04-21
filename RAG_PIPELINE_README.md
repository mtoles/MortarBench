# Fannie Mae Selling Guide RAG Pipeline

This document provides an end-to-end overview, architectural breakdown, and operational guide for `rag_pipeline.py`.

This pipeline implements a state-of-the-art **Retrieval-Augmented Generation (RAG)** system built to accurately answer natural language questions about the complex policies found in the *Fannie Mae Selling Guide* (PDF).

---

## 1. Overview and Architecture

The `rag_pipeline.py` script utilizes the `LangChain` framework. It extracts text from a dense corporate PDF document, converts that text into measurable semantic vectors (embeddings) locally, stores them in a local index, and feeds the most relevant chunks into a highly capable large language model (Llama 3 running locally via Ollama) for final synthesis and answering.

**This entire pipeline is 100% free, running entirely on your local machine with zero API calls.**

### Architectural Best Practices Implemented:

1. **Document Loading (`PyMuPDFLoader`)**:
   Instead of using standard loaders like `PyPDFLoader`, the system leverages PyMuPDF (fitz), which is highly robust for maintaining structural integrity and text flow across complex layouts (such as multi-column corporate PDFs).
2. **Text Chunking (`RecursiveCharacterTextSplitter`)**:
   The Selling Guide is broken down into semantic chunks of **1000 characters** with an **overlap of 200 characters**.

   - **Why?** Breaking on paragraph and sentence boundaries recursively protects sub-clauses from being awkwardly torn apart. A 200-character overlap gives boundary context to the retriever so an LLM doesn't read a sentence halfway.
3. **Embeddings Model (`BAAI/bge-large-en-v1.5`)**:
   The system utilizes open-source HuggingFace embeddings (`BAAI/bge-large-en-v1.5`).

   - **Why?** This model routinely ranks at or near the top of the Massive Text Embedding Benchmark (MTEB) for retrieval tasks, providing extremely high semantic correlation between a user's question and the correct document passage.
4. **Vector Database (`FAISS`)**:
   Chunks are stored locally on-disk using **FAISS** (Facebook AI Similarity Search). This allows instantaneous query ingestion without vector cloud subscription costs.
5. **Retrieval Strategy (`Maximal Marginal Relevance - MMR`)**:
   Instead of basic similarity search, the retriever uses MMR (`k=5`, `fetch_k=20`).

   - **Why?** Basic vector search might return five chunks of text that repeat the exact same piece of information. MMR finds 20 highly relevant chunks, then selects the top 5 that are both highly relevant **and diverse** from one another, ensuring the LLM gets the widest applicable context.
6. **Large Language Model (`Llama 3 via Ollama`)**:
   The ultimate summarizer is the open-source `llama3` model running locally. This ensures 100% data privacy and zero API costs, while still providing robust zero-shot factual answering. It operates under a highly constrained prompt requiring it to pull exclusively from the context instead of hallucinating outside information.

---

## 2. Prerequisites

### Environment Setup

Your project must have the dependencies installed. If not already installed, run:

```bash
pip install langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers pymupdf python-dotenv
```

### Ollama Setup

Because the entire pipeline runs locally (100% free), you must have Ollama installed on your machine to power the LLM server.

1. Download and install [Ollama](https://ollama.com/)
2. Open a terminal and run `ollama run gemma3` to pre-download the Gemma 3 model (this is a one-time download of approx 4.7 GB).

---

## 3. Script Structure

- `load_and_split_documents(pdf_path)`: Uses PyMuPDF to extract all pages from the PDF, passes them to the chunker, and outputs formatted split documents.
- `build_retrievers(splits, force_rebuild)`: Loads the `HuggingFaceEmbeddings`. Either pulls the historical FAISS index directly from the `faiss_index_bge` local directory or constructs a fresh one if `force_rebuild` is used. Applies the MMR retriever logic.
- `create_rag_chain(retriever)`: Initializes the local LLM (`Ollama / Llama 3`) and the strict system prompt. Links them into a LangChain Expression Language (LCEL) chain consisting of:
  `Context Retriever -> Prompt -> LLM -> String Output Parser`
- `main()`: Manages the CLI arguments, initializes the components, and controls interactive loops.

---

## 4. How to Run

There are three primary ways to interact with the script. *(Note: The very first time you execute the script, it will take several minutes to chunk the PDF, calculate all semantic vectors locally, and build the FAISS index folder. Subsequent runs will be instantaneous).*

### A. Run a Single Inquiry

For quick terminal questions and answers:

```bash
python rag_pipeline.py --query "What is the maximum LTV for a cash-out refinance?"
```

### B. Start an Interactive Chat Session

If you omit the `--query` flag, you enter interactive mode where you can ask continuous questions:

```bash
python rag_pipeline.py
```

*Example input:*

```
Interactive RAG session started. Type 'exit' to quit.

Enter your question: What is a large deposit?

Answer:
Based on the provided Fannie Mae Selling Guide, a large deposit...
```

*(To leave the session, type `exit` or `quit`, or press `Ctrl+C`)*

### C. Force an Index Rebuild

If you update the source PDF or change the chunking parameters in the script, you **must** wipe the old FAISS indices and rebuild them.

```bash
python rag_pipeline.py --rebuild
```

You can also append this to a query:

```bash
python rag_pipeline.py --rebuild --query "What are the rules on employment verification?"
```

### D. Evaluate Loans using the RAG Reflection Agent

The RAG pipeline is deeply integrated with the `ReflectionAgent` to provide strict factual grounding when analyzing loan cases against the Fannie Mae Selling Guide. The agent performs initial checks using standard prompt logic, then retrieves context from the FAISS RAG database, and performs a reflective audit on its own preliminary answer.

To test the RAG Reflection agent on a specific example in the benchmark:

```bash
python eval.py --model_id gpt-5 --model_type reflection --trials 1 --row_indexes 0 --use_domain_expertise
```

*(To run the entire benchmarking set, omit the `--row_indexes 0` argument).*

---

## 5. Reflection Agent Integration Architecture

To effectively scale the RAG pipeline to bulk loan evaluation (`eval.py`) without encountering memory bottlenecks from launching multiple asynchronous embeddings models, a singleton architecture is seamlessly utilized:

1. **Singleton Retriever (`get_shared_retriever`)**: The `reflection_agent.py` script initializes the underlying embedding model (`BAAI/bge-large-en-v1.5`) and loads the FAISS index once globally. Concurrent evaluator threads safely share this memory footprint across parallel processes.
2. **Pre-emptive Context Retrieval (`set_context`)**: Instead of fetching RAG data repetitively inside the loop, the query is processed by the RAG index exactly once securely upon loan question initialization, keeping computational overhead aggressively optimized.
3. **Reflective Self-Correction**: The precise chunks outputted from FAISS are strictly embedded directly into the `source_doc` constraint of the subsequent AI pass, explicitly commanding the LLM to overrule any logical errors in its initial reasoning by specifically citing the mandated Fannie Mae Selling Guide regulations.
