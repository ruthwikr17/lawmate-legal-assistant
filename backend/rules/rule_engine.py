# backend/rules/rule_engine.py

import json
from pathlib import Path
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).resolve().parents[2]
RULES_FILE = BASE_DIR / "backend/rules/rules.json"

class LegalRuleEngine:
    def __init__(self):
        self.rules = []
        self._load_rules()

    def _load_rules(self):
        """Loads rules from the JSON configuration."""
        if not RULES_FILE.exists():
            print(f"Warning: {RULES_FILE} not found. Rule engine disabled.")
            return

        try:
            with open(RULES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.rules = data.get("mappings", [])
            print(f"LegalRuleEngine initialized with {len(self.rules)} rules.")
        except Exception as e:
            print(f"Error loading rules: {e}")

    def find_rules(self, query: str) -> List[Dict]:
        """
        Matches a user query against high-impact legal rules.
        Returns a list of matching rules to act as 'Statutory Anchors'.
        """
        query_lower = query.lower()
        matches = []
        
        for rule in self.rules:
            # Simple keyword matching for speed and reliability
            if any(kw in query_lower for kw in rule.get("keywords", [])):
                matches.append({
                    "intent": rule["intent"],
                    "rule": rule["rule"],
                    "description": rule["description"]
                })
        
        return matches

if __name__ == "__main__":
    engine = LegalRuleEngine()
    test_query = "The police took my phone and didn't give a receipt."
    found = engine.find_rules(test_query)
    for r in found:
        print(f"Anchor Found: {r['rule']} | {r['description']}")
