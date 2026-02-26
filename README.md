# LawMate: Jurisdiction-Aware Legal Intelligence

(In progress) <br>

> LawMate is a Retrieval-Augmented Generation (RAG) pipeline designed to democratize Indian legal knowledge. It bridges the gap between complex statutory language and the everyday citizen by transforming raw legal texts into structured, plain-language, and highly actionable advice.  

Unlike generic AI chatbots that often hallucinate or provide geographically inaccurate advice, LawMate enforces jurisdiction-aware retrieval and a statute-first reasoning engine to ensure users receive accurate guidance tailored to their specific state or city.

## Core Capabilities
**Hierarchical Jurisdiction Filtering:** Automatically maps user queries to the correct municipal, state, or central statutes, preventing cross-jurisdiction contamination.

**Statute-First Reasoning:** Prioritizes statutory law to define absolute rights, utilizing case law strictly for illustrative application.

**Actionable Output Generation:** Bypasses dense academic explanations in favor of tactical next steps, clear power dynamics, and structured procedural guidance.

**Domain Versatility:** Currently handles rent control, tenant eviction, motor accident claims (MACT), alimony disputes, and institutional fee disputes.


## System Architecture
The project operates on a Directed Acyclic Graph (DAG) pipeline, utilizing the Gemini API for the LLM layer and ChromaDB for vector storage. <br>

1. **Ingestion & Parsing:** Extracts and cleans text from central, state, and municipal PDFs.

2. **Semantic Chunking:** Splits text while preserving critical metadata (Document Type, Jurisdiction Level, State).

3. **Vector Indexing:** Embeds chunks using all-mpnet-base-v2 into a persistent ChromaDB instance.

4. **Hybrid Retrieval:** Re-ranks results based on vector similarity, keyword overlap, and source diversity, heavily weighted by jurisdiction.

5. **Generation Layer:** Orchestrates the LLM to output litigation-aware, highly structured responses devoid of standard AI boilerplate.


## Tech Stack
**Core:** Python 3.10+

**Embeddings:** SentenceTransformers (all-mpnet-base-v2)

**Vector Store:** ChromaDB

**LLM Orchestration:** Gemini API (gemini-2.5-flash)

**Document Processing:** PyMuPDF / fitz


## Project Structure

```
lawmate/
├── db/                             # Persistent local storage for the ChromaDB vector index
├── storage/
│   └── processed_chunks/           # Directory for semantic text blocks and metadata
├── scripts/
│   ├── ingest_documents.py         # Handles raw PDF parsing and text cleaning
│   ├── build_chunks.py             # Executes semantic splitting and metadata tagging
│   ├── chunk_distribution_check.py # Validates corpus balance and document type distribution
│   ├── index_chroma.py             # Manages embedding and vector database indexing
│   └── pipeline/
│       ├── retrieve.py             # Executes the hierarchical, jurisdiction-aware search logic
│       └── generate_response.py    # Manages LLM prompting, tone constraints, and structured output
└── README.md
```

## Roadmap & Future Extensions
The system is actively being refined by Ruthvik at Sphoorthy Engineering College to shift from a functional prototype to a comprehensive legal assistant. Immediate priorities include:

- **Advanced Response Calibration:** Tuning the LLM for higher authority, enforcing "burden of proof" framing, and eliminating AI-style transitions.

- **Workflow Automation Modes:** Adding guided, step-by-step procedures for drafting consumer complaints, FIRs, and MACT claims.

- **Confidence Scoring:** Implementing a dual-score system for retrieval accuracy and legal coverage confidence.

- **Ecosystem Expansion:** Scaling the corpus to encompass all Indian states, adding multi-lingual support (Telugu/Hindi), and migrating to a Streamlit web interface.


## Disclaimer
LawMate is an AI-based informational prototype developed for research and educational purposes. It does not constitute formal legal counsel and does not replace the advice of a licensed advocate. Users should consult qualified legal professionals before initiating any litigation or formal legal proceedings.