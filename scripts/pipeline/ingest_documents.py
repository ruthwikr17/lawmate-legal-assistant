# scripts/pipeline/ingest_documents.py

# Read PDFs from storage/raw_documents/
# Extract text
# Clean it
# Save plain text output in JSON


from pathlib import Path
import pdfplumber
import fitz
import json
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

INPUT_DIR = BASE_DIR / "storage/raw_documents"
OUTPUT_DIR = BASE_DIR / "storage/processed_chunks/cleaned"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    text = ""

    # Try pdfplumber first
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if text.strip():
            return text

    except Exception:
        pass

    # Fallback to PyMuPDF
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()

    except Exception as e:
        print(f"Both extraction methods failed for {pdf_path}: {e}")
        return ""

    return text


def ingest_documents():
    for pdf_file in INPUT_DIR.rglob("*"):
        if pdf_file.is_file() and pdf_file.suffix.lower() == ".pdf":
            relative_path = pdf_file.relative_to(INPUT_DIR)
            output_file = OUTPUT_DIR / relative_path.with_suffix(".json")

            # SKIP if already processed
            if output_file.exists():
                continue

            print(f"Processing: {pdf_file}")

            output_file.parent.mkdir(parents=True, exist_ok=True)

            text = extract_text_from_pdf(pdf_file)

            if not text.strip():
                print(f"Skipping empty or failed file: {pdf_file}")
                continue

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"text": text, "source_path": str(pdf_file)}, f, indent=2)

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_documents()
