# backend/retrieval/retriever.py

import re
import os
import json
import chromadb
import sys
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ---------------- CONFIG ---------------- #
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from backend.llm.llm_utils import LLMService

PERSIST_DIR = BASE_DIR / "db"
CHUNKS_FILE = BASE_DIR / "storage/processed_chunks/chunks.jsonl"
COLLECTION_NAME = "lawmate_collection"

class LegalRetriever:
    def __init__(self):
        print("Initializing LegalRetriever (Phase 1 Optimized)...")
        try:
            # Bi-Encoder for initial retrieval
            self.bi_encoder = SentenceTransformer("all-mpnet-base-v2")
                   
            # Cross-Encoder for high-precision reranking (Free & Local)
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            
            # LLM Service for complex logic
            self.llm_service = LLMService()
            
            # ChromaDB client
            self.client = chromadb.Client(
                Settings(persist_directory=str(PERSIST_DIR), is_persistent=True)
            )
            self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
            
            # Load chunks for BM25
            self.chunks = []
            self.bm25 = None
            self._load_bm25_index()
        except Exception as e:
            print(f"CRITICAL ERROR: LegalRetriever failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def _load_bm25_index(self):
        """Loads chunks from JSONL and initializes BM25 index."""
        if not CHUNKS_FILE.exists():
            print(f"Warning: {CHUNKS_FILE} not found. BM25 will be disabled.")
            return

        print("Loading chunks for BM25 index...")
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))
        
        # Tokenize for BM25
        tokenized_corpus = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index initialized with {len(self.chunks)} chunks.")

    def hybrid_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Applies Cross-Encoder reranking to top candidates.
        """
        if not candidates:
            return []

        # Deduplicate candidates by text
        seen_texts = set()
        unique_candidates = []
        for c in candidates:
            if c["text"] not in seen_texts:
                unique_candidates.append(c)
                seen_texts.add(c["text"])

        # Prepare pairs for Cross-Encoder
        pairs = [[query, c["text"]] for c in unique_candidates]
        cross_scores = self.cross_encoder.predict(pairs)
        
        for i, candidate in enumerate(unique_candidates):
            candidate["rerank_score"] = float(cross_scores[i])
            
            # Boost specific doc types
            doc_type = candidate["metadata"].get("document_type")
            if doc_type in ["statute", "constitution"]:
                candidate["rerank_score"] += 0.15 # Stronger boost for primary sources
            
        # Sort by rerank score descending
        unique_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Diversity Control
        diversified = []
        source_counter = {}
        for res in unique_candidates:
            source = res["metadata"].get("source_file")
            if source_counter.get(source, 0) < 2:
                diversified.append(res)
                source_counter[source] = source_counter.get(source, 0) + 1
            
            if len(diversified) >= 6: # Return top 6 high-quality results
                break
                
        return diversified

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        print(f"Executing retrieval for: '{query}'")
        
        # 1. LLM-based Jurisdiction Detection
        jurisdiction = self.llm_service.detect_jurisdiction(query)
        level = jurisdiction.get("level")
        region = jurisdiction.get("region")
        print(f"Detected Jurisdiction: {level} ({region})")

        # 2. Vector Search (Semantic)
        query_embedding = self.bi_encoder.encode(query).tolist()
        
        # Filter logic: if city or state, try to filter first
        where_filter = None
        if level in ["city", "state"] and region:
            where_filter = {"state": region}

        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        candidates = []
        if vector_results["documents"]:
            for i in range(len(vector_results["documents"][0])):
                candidates.append({
                    "text": vector_results["documents"][0][i],
                    "metadata": vector_results["metadatas"][0][i],
                    "search_type": "vector"
                })

        # 3. BM25 Search (Keyword) - helpful for specific legal terms/sections
        if self.bm25:
            # Clean query for BM25
            tokenized_query = re.sub(r"[^\w\s]", "", query).lower().split()
            bm25_hits = self.bm25.get_top_n(tokenized_query, self.chunks, n=top_k)
            for cand in bm25_hits:
                # Basic metadata filter check
                if region and cand["metadata"].get("state") != region:
                    continue
                
                candidates.append({
                    "text": cand["text"],
                    "metadata": cand["metadata"],
                    "search_type": "bm25"
                })

        # 4. Global Fallback if filter result is too small
        if len(candidates) < 5 and where_filter:
            print("Insufficient local results, falling back to global search...")
            fallback_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            for i in range(len(fallback_results["documents"][0])):
                candidates.append({
                    "text": fallback_results["documents"][0][i],
                    "metadata": fallback_results["metadatas"][0][i],
                    "search_type": "fallback"
                })

        # 5. Rerank (Cross-Encoder)
        return self.hybrid_rerank(query, candidates)

if __name__ == "__main__":
    retriever = LegalRetriever()
    queries = [
        "What are my rights against sudden eviction in Hyderabad?",
        "How to file a consumer complaint for a faulty washing machine?",
        "Alimony laws in India for a working wife"
    ]
    
    for q in queries:
        print("\n" + "="*50)
        hits = retriever.retrieve(q)
        for h in hits:
            print(f"[{h.get('search_type', 'unknown').upper()}] {h['metadata'].get('document_title')} | Score: {h['rerank_score']:.4f}")
            print(f"{h['text'][:150]}...\n")
