# backend/workflows/workflow_engine.py

import json
from pathlib import Path
from typing import List, Optional

class WorkflowEngine:
    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            data_path = Path(__file__).parent / "workflows.json"
        else:
            data_path = Path(data_path)
        self.data_path = data_path
        self._workflows = []
        self.load_workflows()

    def load_workflows(self):
        try:
            with open(self.data_path, 'r') as f:
                self._workflows = json.load(f)
        except Exception as e:
            print(f"Error loading workflows: {e}")
            self._workflows = []

    def get_all_workflows(self) -> List[dict]:
        return self._workflows

    def get_workflow_by_id(self, workflow_id: str) -> Optional[dict]:
        for w in self._workflows:
            if w["id"] == workflow_id:
                return w
        return None

    def search_workflows(self, query: str) -> List[dict]:
        query = query.lower()
        results = []
        for w in self._workflows:
            if query in w["title"].lower() or query in w["desc"].lower():
                results.append(w)
        return results
