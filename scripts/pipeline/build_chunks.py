# scripts/pipeline/build_chunks.py

# Split long documents into chunks
# Generate metadata (important for jurisdiction filtering)
# Write .jsonl for indexing


import json
import re
import uuid
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

CLEANED_DIR = BASE_DIR / "storage/processed_chunks/cleaned"
OUTPUT_FILE = BASE_DIR / "storage/processed_chunks/chunks.jsonl"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ".", " "]
)


def extract_year_from_filename(filename: str):
    match = re.search(r"(19|20)\d{2}", filename)
    return int(match.group()) if match else None


def detect_document_type(file_path: Path):
    path_str = str(file_path).lower()

    if "case_law" in path_str:
        return "case_law"
    if "constitution" in path_str:
        return "constitution"
    return "statute"


def detect_jurisdiction(file_path: Path):
    parts = [p.lower() for p in file_path.parts]

    if "metropolitan_cities" in parts:
        idx = parts.index("metropolitan_cities")
        if len(parts) > idx + 1:
            return "city", parts[idx + 1].title(), None

    if "state" in parts:
        idx = parts.index("state")
        if len(parts) > idx + 1:
            return "state", parts[idx + 1].title(), None

    if "central" in parts:
        return "central", None, None

    if "case_law" in parts:
        return "central", None, "Supreme Court"

    return "unknown", None, None


def build_chunks():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for file in CLEANED_DIR.rglob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            text = data.get("text", "").strip()
            if not text:
                continue

            document_type = detect_document_type(file)
            jurisdiction_level, state_name, court = detect_jurisdiction(file)

            original_pdf = file.with_suffix(".pdf").name
            year = extract_year_from_filename(original_pdf)

            chunks = splitter.split_text(text)

            for chunk in chunks:
                chunk_id = str(uuid.uuid4())

                metadata = {
                    "document_type": document_type,
                    "jurisdiction_level": jurisdiction_level,
                    "state": state_name,
                    "court": court,
                    "year": year,
                    "source_file": original_pdf,
                }

                record = {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": metadata,
                }

                outfile.write(json.dumps(record) + "\n")
                total_chunks += 1

    print(f"Chunking complete. Total chunks created: {total_chunks}")


if __name__ == "__main__":
    build_chunks()
