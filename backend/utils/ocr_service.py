# backend/utils/ocr_service.py

import easyocr
import cv2
import numpy as np
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict

class OCRService:
    def __init__(self):
        # Initialize reader (English as default)
        print("Initializing OCR (EasyOCR)...")
        self.reader = easyocr.Reader(['en'], gpu=False)

    def extract_text(self, file_bytes: bytes) -> str:
        """
        Extracts text from raw bytes (Image or PDF).
        """
        try:
            # 1. More robust PDF detection (check first 1024 bytes)
            is_pdf = b"%PDF-" in file_bytes[:1024]
            
            if is_pdf:
                print("DEBUG: Detected PDF file.")
                return self._extract_from_pdf(file_bytes)
            
            # 2. Otherwise treat as Image
            print("DEBUG: Treating file as Image.")
            return self._extract_from_image(file_bytes)
            
        except Exception as e:
            print(f"CRITICAL EXTRACTION ERROR: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _extract_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extracts text from PDF (Searchable or Scanned)."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = []
        
        for page in doc:
            text = page.get_text().strip()
            if text:
                full_text.append(text)
            else:
                # Scanned PDF Fallback: Render to Image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # High-res
                img_bytes = pix.tobytes("png")
                ocr_text = self._extract_from_image(img_bytes)
                if ocr_text:
                    full_text.append(ocr_text)
                    
        return "\n".join(full_text).strip()

    def _extract_from_image(self, image_bytes: bytes) -> str:
        """Extracts text from raw image bytes using EasyOCR."""
        results = self.reader.readtext(image_bytes)
        return " ".join([res[1] for res in results]).strip()

if __name__ == "__main__":
    pass
