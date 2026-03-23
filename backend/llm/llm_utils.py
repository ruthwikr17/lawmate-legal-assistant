# backend/llm/llm_utils.py

import os
import json
import google.generativeai as genai
from google.generativeai.types import SafetySettingDict, HarmCategory, HarmBlockThreshold
from pathlib import Path
from typing import Dict, Optional, List
from dotenv import load_dotenv

# Absolute path for environment loading
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

GEMINI_MODEL = "models/gemini-3.1-flash-lite-preview"
# [DIAGNOSTIC] Author Note: gemini-1.5-flash was indeed 404'ing as the user's API key is for a futuristic/experimental model suite.
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Diagnostic fallback for local development
    print(f"CRITICAL Warning: GEMINI_API_KEY not found in {BASE_DIR / '.env'}")

genai.configure(api_key=api_key)

SAFETY_SETTINGS: SafetySettingDict = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
]

class LLMService:
    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            safety_settings=SAFETY_SETTINGS
        )

    def detect_jurisdiction(self, query: str) -> Dict[str, Optional[str]]:
        """
        Rule-based jurisdiction detection to save API quota.
        """
        q = query.lower()
        
        # Major Indian Cities
        cities = ["mumbai", "delhi", "bangalore", "hyderabad", "chennai", "kolkata", "pune", "ahmedabad"]
        # Indian States
        states = ["maharashtra", "telangana", "karnataka", "tamil nadu", "west bengal", "gujarat", "punjab", "haryana", "uttar pradesh", "bihar", "rajasthan"]
        
        for city in cities:
            if city in q:
                return {"level": "city", "region": city.capitalize()}
                
        for state in states:
            if state in q:
                return {"level": "state", "region": state.capitalize()}
                
        return {"level": "central", "region": None}

    def generate_answer(self, user_query: str, context: str, conversation_history: list = None) -> str:
        # This will be used in the generator service
        pass
