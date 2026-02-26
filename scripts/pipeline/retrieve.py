# scripts/pipeline/retrieve.py

# Connect to persistent ChromaDB (db/)
# Load indexed legal document collection
# Classify legal domain of user query using LLM (deterministic, rewrite-only)
# Generate dual embeddings (original query + domain phrase)
# Fuse embeddings using weighted vector combination
# Detect jurisdiction level (city / state / central)
# Perform hierarchical metadata-filtered retrieval
# Retrieve top candidate chunks (vector similarity search)
# Apply hybrid reranking (vector score + keyword overlap + statute bias)
# Enforce document-level diversity (max 1 chunk per source file)
# Return structured top results:
#   - Top 3 statutes (from different documents)
#   - Top 2 case laws (from different documents)
# Print ranked results with metadata


import re
import sys
import requests
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from pathlib import Path
from generate_response import generate_answer


model = SentenceTransformer("all-mpnet-base-v2")

# ---------------- CONFIG ---------------- #

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

PERSIST_DIR = BASE_DIR / "db"
COLLECTION_NAME = "lawmate_collection"

ALPHA = 0.75  # weight for original query (domain gets 0.25)

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

# ----------- OLLAMA CONFIG ----------- #

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"

# ---------------- DOMAIN CLASSIFICATION ---------------- #


def classify_domain_phrase(query: str):

    prompt = f"""
        Classify the legal domain of the query.

        Return only a short legal domain phrase.
        2 to 6 words maximum.
        Lowercase.
        No statute names.
        No years.
        No explanation.

        Query: "{query}"
    """

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=10,
        )

        raw = response.json()["response"].strip().lower()

        # Clean aggressively
        cleaned = re.sub(r"[^a-z\s]", "", raw)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        words = cleaned.split()

        # Relax constraints
        if 1 <= len(words) <= 6:
            return " ".join(words[:6])

    except Exception:
        pass

    return None


# ---------------- JURISDICTION DETECTION ---------------- #


def detect_query_jurisdiction(query: str):
    q = query.lower()

    cities = ["hyderabad", "mumbai", "bengaluru", "chennai", "kolkata", "delhi"]
    for city in cities:
        if city in q:
            return "city", city.title()

    states = [
        "telangana",
        "karnataka",
        "maharashtra",
        "tamil nadu",
        "kerala",
        "andhra pradesh",
    ]
    for state in states:
        if state in q:
            return "state", state.title()

    return "central", None


# ---------------- EMBEDDING FUSION ---------------- #


def build_fused_embedding(query: str):

    domain_phrase = classify_domain_phrase(query)
    print("Detected Domain Phrase:", domain_phrase)

    v_query = model.encode(query, normalize_embeddings=True)

    if domain_phrase:
        v_domain = model.encode(domain_phrase, normalize_embeddings=True)
        v_final = ALPHA * v_query + (1 - ALPHA) * v_domain
    else:
        v_final = v_query

    return v_final.tolist()


# ---------------- KEYWORD OVERLAP ---------------- #


def keyword_overlap_score(query, text):
    query_tokens = set(re.findall(r"\w+", query.lower()))
    text_tokens = set(re.findall(r"\w+", text.lower()))
    overlap = query_tokens.intersection(text_tokens)
    return len(overlap) / (len(query_tokens) + 1)


# ---------------- HYBRID RERANK ---------------- #


def hybrid_rerank(query, results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    scored_results = []

    for i in range(len(documents)):
        doc = documents[i]
        metadata = metadatas[i]
        distance = distances[i]

        # --- Semantic Score ---
        semantic_score = 1 - distance

        # --- Keyword Overlap ---
        keyword_score = keyword_overlap_score(query, doc)

        # --- Neutral Source Hierarchy Weight ---
        doc_type = metadata.get("document_type")

        if doc_type == "statute":
            type_weight = 0.08  # small universal boost
        elif doc_type == "constitution":
            type_weight = 0.06
        elif doc_type == "case_law":
            type_weight = 0.02
        else:
            type_weight = 0.0

        # --- Final Score ---
        final_score = 0.65 * semantic_score + 0.25 * keyword_score + type_weight

        scored_results.append((final_score, doc, metadata))

    # Sort by final score descending
    scored_results.sort(reverse=True, key=lambda x: x[0])

    # --- Diversity Control ---
    diversified = []
    source_counter = {}

    for score, doc, metadata in scored_results:
        source = metadata.get("source_file")

        if source not in source_counter:
            source_counter[source] = 0

        if source_counter[source] < 2:
            diversified.append((score, doc, metadata))
            source_counter[source] += 1

        if len(diversified) == 5:
            break

    return diversified


# ---------------- HIERARCHICAL RETRIEVAL ---------------- #


def hierarchical_query(collection, query, fused_embedding):

    level, region = detect_query_jurisdiction(query)
    print(f"Detected jurisdiction level: {level}, region: {region}")

    CITY_TO_STATE = {
        "Hyderabad": "Telangana",
        "Mumbai": "Maharashtra",
        "Chennai": "Tamil Nadu",
        "Bengaluru": "Karnataka",
        "Kolkata": "West Bengal",
        "Delhi": "Delhi",
    }

    state_region = None

    if level == "city":
        state_region = CITY_TO_STATE.get(region)
    elif level == "state":
        state_region = region

    final_results = []

    # --------------------------------------------------
    # 1. STATE STATUTES (Primary if region detected)
    # --------------------------------------------------

    if state_region:
        state_filter = {
            "$and": [{"jurisdiction_level": "state"}, {"state": state_region}]
        }

        state_statutes = collection.query(
            query_embeddings=[fused_embedding],
            n_results=15,
            where=state_filter,
            include=["documents", "metadatas", "distances"],
        )

        if state_statutes["documents"][0]:
            top_state_statutes = hybrid_rerank(query, state_statutes)
            final_results.extend(top_state_statutes[:3])

    # --------------------------------------------------
    # 2. CENTRAL STATUTES (Fallback / Supplement)
    # --------------------------------------------------

    central_filter = {
        "$and": [{"jurisdiction_level": "central"}, {"document_type": "statute"}]
    }

    central_statutes = collection.query(
        query_embeddings=[fused_embedding],
        n_results=10,
        where=central_filter,
        include=["documents", "metadatas", "distances"],
    )

    if central_statutes["documents"][0]:
        top_central_statutes = hybrid_rerank(query, central_statutes)
        final_results.extend(top_central_statutes[:2])

    # --------------------------------------------------
    # 3. CASE LAW (Always Secondary)
    # --------------------------------------------------

    case_filter = {"document_type": "case_law"}

    case_results = collection.query(
        query_embeddings=[fused_embedding],
        n_results=10,
        where=case_filter,
        include=["documents", "metadatas", "distances"],
    )

    if case_results["documents"][0]:
        top_cases = hybrid_rerank(query, case_results)
        final_results.extend(top_cases[:2])

    # --------------------------------------------------
    # 4. If still empty → Global fallback
    # --------------------------------------------------

    if not final_results:
        global_results = collection.query(
            query_embeddings=[fused_embedding],
            n_results=15,
            include=["documents", "metadatas", "distances"],
        )

        final_results = hybrid_rerank(query, global_results)

    # Limit total results
    return final_results[:5]


# ---------------- MAIN ---------------- #


def query_lawmate(user_query):

    client = chromadb.Client(
        Settings(persist_directory=str(PERSIST_DIR), is_persistent=True)
    )

    collection = client.get_collection(name=COLLECTION_NAME)

    fused_embedding = build_fused_embedding(user_query)

    results = hierarchical_query(collection, user_query, fused_embedding)

    answer = generate_answer(user_query, results)

    print("\nResponse:")
    print("\n" + "-" * 80)
    print(answer)
    print("-" * 80 + "\n")


if __name__ == "__main__":
    query = input("Enter legal query: ")
    query_lawmate(query)
