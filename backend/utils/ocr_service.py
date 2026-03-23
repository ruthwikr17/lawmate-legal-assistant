# backend/utils/ocr_service.py

import easyocr
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict

class OCRService:
    def __init__(self):
        # Initialize reader (English as default for legal docs in India)
        # gpu=False for compatibility, but can be enabled if user has GPU
        print("Initializing OCR (EasyOCR)...")
        self.reader = easyocr.Reader(['en'], gpu=False)

    def extract_text_from_image(self, image_input) -> str:
        """
        Extracts text from an image (bytes or path).
        """
        try:
            results = self.reader.readtext(image_input)
            # Combine text results
            extracted_text = " ".join([res[1] for res in results])
            return extracted_text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    def analyze_document_risk(self, text: str) -> Dict:
        """
        Basic risk analysis for legal documents.
        (More advanced logic can be added in Phase 4)
        """
        risk_keywords = {
            "sudden_eviction": ["eviction", "notice", "vacate", "terminate"],
            "financial_penalty": ["penalty", "fine", "forfeit", "interest"],
            "unfair_terms": ["unilateral", "sole discretion", "non-refundable"]
        }
        
        detected_risks = []
        text_lower = text.lower()
        
        for risk, keywords in risk_keywords.items():
            found = [k for k in keywords if k in text_lower]
            if found:
                detected_risks.append({"type": risk, "keywords": found})
                
        return {
            "original_text": text,
            "detected_risks": detected_risks,
            "summary": "Document processed and scanned for risk markers."
        }

if __name__ == "__main__":
    # Test OCR (requires sample image)
    pass
