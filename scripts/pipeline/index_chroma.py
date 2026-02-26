# scripts/pipeline/index_chroma.py

# Reads JSONL
# Connects to persistent ChromaDB
# Embeds chunks (SentenceTransformer)
# Inserts in batches
# Stores in db/


import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

CHUNKS_FILE = BASE_DIR / "storage/processed_chunks/chunks.jsonl"
PERSIST_DIR = BASE_DIR / "db"
COLLECTION_NAME = "lawmate_collection"

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)


def index_chunks():

    client = chromadb.Client(
        Settings(persist_directory=str(PERSIST_DIR), is_persistent=True)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    batch_ids = []
    batch_texts = []
    batch_metadata = []

    batch_size = 100
    total = 0

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            batch_ids.append(record["id"])
            batch_texts.append(record["text"])

            clean_metadata = {
                k: v for k, v in record["metadata"].items() if v is not None
            }

            batch_metadata.append(clean_metadata)

            if len(batch_ids) >= batch_size:
                collection.add(
                    ids=batch_ids, documents=batch_texts, metadatas=batch_metadata
                )

                total += len(batch_ids)
                print(f"Indexed {total} chunks...")

                batch_ids, batch_texts, batch_metadata = [], [], []

        if batch_ids:
            collection.add(
                ids=batch_ids, documents=batch_texts, metadatas=batch_metadata
            )
            total += len(batch_ids)

    print(f"Indexing complete. Total indexed: {total}")


if __name__ == "__main__":
    index_chunks()
