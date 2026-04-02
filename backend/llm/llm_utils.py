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
        Rock-solid jurisdiction detection.
        """
        q = query.lower().strip()
        
        # Major Indian Cities & States (Expanded)
        cities = {
            "mumbai": "Mumbai", "bombay": "Mumbai",
            "delhi": "Delhi", "ncr": "Delhi", "noida": "Delhi", "gurgaon": "Delhi",
            "bangalore": "Bangalore", "bengaluru": "Bangalore",
            "hyderabad": "Hyderabad",
            "chennai": "Chennai", "madras": "Chennai",
            "kolkata": "West Bengal", "calcutta": "West Bengal",
            "pune": "Pune",
            "ahmedabad": "Gujarat"
        }
        states = {
            "maharashtra": "Maharashtra", "telangana": "Telangana",
            "karnataka": "Karnataka", "tamil nadu": "Tamil Nadu",
            "west bengal": "West Bengal", "gujarat": "Gujarat",
            "punjab": "Punjab", "haryana": "Haryana",
            "uttar pradesh": "Uttar Pradesh", "up": "Uttar Pradesh",
            "bihar": "Bihar", "rajasthan": "Rajasthan", "kerala": "Kerala"
        }
        
        # Check for State matches first (broader)
        for s_key, s_name in states.items():
            if s_key in q:
                return {"level": "state", "region": s_name}
                
        # Check for City matches
        for c_key, c_name in cities.items():
            if c_key in q:
                # Map some cities directly to their states for better retrieval filtering
                state_map = {"Mumbai": "Maharashtra", "Pune": "Maharashtra", "Bangalore": "Karnataka", "Hyderabad": "Telangana", "Chennai": "Tamil Nadu"}
                region = state_map.get(c_name, c_name)
                return {"level": "city" if c_name not in ["West Bengal", "Gujarat"] else "state", "region": region}
                
        return {"level": "central", "region": None}

    def generate_answer(self, user_query: str, context: str, conversation_history: List = None) -> str:
        # This is implemented in the generator service
        return ""
