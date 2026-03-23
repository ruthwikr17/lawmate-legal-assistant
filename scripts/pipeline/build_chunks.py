# scripts/pipeline/build_chunks.py

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

# Standard splitter for sub-chunking if a section is too large
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " "]
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


def split_by_sections(text: str):
    """
    Splits text by Section headers like '1. Short title...', '2. Definitions.'
    Returns a list of (section_title, section_text)
    """
    # Pattern for section start: digit followed by dot and space, e.g., "1. "
    # We look for it at the start of a line to avoid false positives middle of text
    pattern = r"(?m)^(\d+)\.\s+([A-Z][a-z].*?)\.\u2014"
    
    sections = []
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return [("General", text)]
    
    # Text before first section
    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.append(("Preamble", preamble))
        
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        
        section_num = matches[i].group(1)
        section_title = matches[i].group(2)
        full_title = f"Section {section_num}: {section_title}"
        section_text = text[start:end].strip()
        
        sections.append((full_title, section_text))
        
    return sections


def build_chunks():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    
    # Get document title from filename for context
    # e.g. consumer_protection_act_2019 -> Consumer Protection Act 2019
    def clean_title(name: str):
        return name.replace("_", " ").replace(".json", "").title()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for file in CLEANED_DIR.rglob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            text = data.get("text", "").strip()
            if not text:
                continue

            doc_title = clean_title(file.name)
            document_type = detect_document_type(file)
            jurisdiction_level, state_name, court = detect_jurisdiction(file)

            original_pdf = file.with_suffix(".pdf").name
            year = extract_year_from_filename(original_pdf)

            # 1. Structural Splitting by Sections
            sections = split_by_sections(text)
            
            for section_title, section_text in sections:
                # 2. Sub-splitting if section is too large
                raw_chunks = recursive_splitter.split_text(section_text)
                
                for chunk in raw_chunks:
                    # Prepend Context for better retrieval
                    # [Document Title] [Section Title]: [Content]
                    enhanced_chunk = f"[{doc_title}] {section_title}\n{chunk}"
                    
                    chunk_id = str(uuid.uuid4())
                    metadata = {
                        "document_title": doc_title,
                        "section_title": section_title,
                        "document_type": document_type,
                        "jurisdiction_level": jurisdiction_level,
                        "state": state_name,
                        "court": court,
                        "year": year,
                        "source_file": original_pdf,
                    }

                    record = {
                        "id": chunk_id,
                        "text": enhanced_chunk,
                        "metadata": metadata,
                    }

                    outfile.write(json.dumps(record) + "\n")
                    total_chunks += 1

    print(f"Structural Chunking complete. Total chunks created: {total_chunks}")


if __name__ == "__main__":
    build_chunks()
