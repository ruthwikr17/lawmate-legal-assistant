# LawMate: Clinical Legal Intelligence for the Everyday Citizen

> **LawMate** is a high-precision, jurisdiction-aware legal assistant designed to democratize access to the Indian Legal System. By combining structural RAG retrieval with a high-speed Hybrid Rule Engine, LawMate transforms complex statutory language into tactical, plain-English guidance at a **10th-grade reading level**.

---

## 🏛️ Core Architectural Pillars

### 1. Hybrid Rule Engine (Statutory Anchors)
LawMate identifies high-impact legal "intents" (Arrest, FIR, Fraud, Eviction) and anchors them to hardcoded, verified statutory rules (BNS 318, BNSS 173). This provides an absolute "Ground Truth" compass before the AI even begins its creative reasoning.

### 2. Advanced RAG (Precision Retrieval)
- **Structural Section Chunking**: Documents are split by Section headers (e.g., "Section 420") rather than random tokens to preserve legal context.
- **Hybrid Retrieval**: Combines BM25 (Keyword) + Sentence-Transformers (Semantic) search.
- **Cross-Encoder Reranking**: Utilizes `ms-marco-MiniLM` to rerank results based on precise relevance.

### 3. Absolute Clarity Protocol (V63)
LawMate is strictly tuned to a **10th-grade reading level**. It explicitly bans academic legal jargon ("indefensible", "void") in favor of everyday language that anyone can understand and act upon.

### 4. Jurisdiction-Aware Logic
A dedicated detection layer extracts city/state context from user queries and applies metadata filters to the vector store, preventing "legal contamination" from irrelevant state laws.

---

## ⚡ Key Features

- **Ask AI**: Natural Language chat with conditional case-based reasoning.
- **Legal Workflows**: Step-by-step procedural manifests for filing FIRs, Consumer Complaints, and more.
- **Rights Awareness**: Interactive educational modules explaining fundamental rights with click-to-explore depth.
- **Document Risk Analysis**: Integrated OCR service that extracts text from legal notices and identifies immediate risks and paper-trail gaps.

---

## 🛠️ Tech Stack

### Frontend (User Interface)
- **Framework**: Next.js 14+ (App Router)
- **Styling**: Vanilla CSS + Tailwind
- **Animations**: Framer Motion
- **Icons**: Lucide React

### Backend (AI & Logic)
- **Core API**: FastAPI (Python)
- **Vector Database**: ChromaDB (with MPNet-V2 Embeddings)
- **LLM**: Gemini-1.5-Flash (Optimized for JSON-Architect logic)
- **Rule Engine**: Custom Keyword-Intent Hybrid Mapper

### Java Backend (Enterprise Context)
- **Architecture**: Spring Boot (used for service orchestration and user rights data management).

---

## 📁 Repository Structure

```text
lawmate/
├── backend/            # Python FastAPI service (RAG, LLM, Rule Engine)
│   ├── rules/          # Statutory Anchor configurations
│   ├── retrieval/      # Hybrid Search & Reranking logic
│   └── workflows/      # Procedural Manifest JSONs
├── frontend/           # Next.js web application
├── java-backend/       # Spring Boot enterprise layer
├── scripts/            # Data pipeline (PDF -> Chunks -> Vector Store)
└── storage/            # Processed legal corpus (JSONL)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- Maven 3.8+ (for Java components)
- `GEMINI_API_KEY` in root `.env`

### Installation
1. **Python AI Service**:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --port 8000
   ```

2. **Java Backend**:
   ```bash
   cd java-backend
   mvn spring-boot:run
   ```

3. **Frontend**:
   ```bash
   cd frontend
   npm install && npm run dev
   ```

---

## ⚖️ Disclaimer
LawMate is an AI-based informational platform for educational purposes. It does **not** constitute formal legal counsel. Users should consult a licensed advocate before initiating any legal proceedings.

---
*Developed by Ruthvik and LawMate contributors to make the Law accessible to everyone.*