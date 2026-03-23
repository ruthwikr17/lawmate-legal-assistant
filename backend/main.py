# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple
import uvicorn
import os
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")
print(f"DEBUG: GEMINI_API_KEY present: {'Yes' if os.getenv('GEMINI_API_KEY') else 'No'}")

from backend.retrieval.retriever import LegalRetriever
from backend.llm.generator import LegalGenerator
from backend.rules.rule_engine import LegalRuleEngine

app = FastAPI(title="LawMate AI Service", version="2.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import FastAPI, HTTPException, UploadFile, File
from backend.utils.ocr_service import OCRService

# Initialize Services
retriever = LegalRetriever()
generator = LegalGenerator()
rule_engine = LegalRuleEngine()
ocr_service = OCRService()

from backend.workflows.workflow_engine import WorkflowEngine
workflow_engine = WorkflowEngine()

@app.get("/workflows")
async def get_workflows():
    workflow_engine.load_workflows() # Force Sync for Dev
    return workflow_engine.get_all_workflows()

@app.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    workflow = workflow_engine.get_workflow_by_id(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow

@app.post("/analyze-doc")
async def analyze_doc(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        extracted_text = ocr_service.extract_text_from_image(contents)
        
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract text from document.")
            
        analysis = ocr_service.analyze_document_risk(extracted_text)
        # Map summary to analysis key for frontend consistency
        return {
            "analysis": analysis["summary"],
            "detected_risks": analysis["detected_risks"],
            "original_text": analysis["original_text"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = None
    context: Optional[dict] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

@app.get("/")
async def root():
    return {"message": "LawMate AI Service is online"}

@app.post("/generate")
async def generate(request: ChatRequest):
    try:
        # 1. Check for Statutory Anchors
        rule_context = rule_engine.find_rules(request.query)
        
        # 2. Retrieve context for precision (V28)
        hits = retriever.retrieve(request.query)
        
        # 3. Generate with architectural constraints and rule anchors
        answer, _ = generator.generate_response(
            request.query,
            hits,
            request.history,
            context=request.context,
            rule_context=rule_context
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # 1. Condense Query if history exists
        processed_query = request.query
        if request.history and len(request.history) > 0:
            try:
                # Ask LLM to rephrase query for retrieval based on history
                history_summary = "\n".join([f"{m['role']}: {m['content']}" for m in request.history[-5:]])
                condense_prompt = f"""
                Given the following conversation history and the latest user response, rephrase the latest response to be a standalone, highly descriptive legal search query for an Indian law database.
                
                RULES:
                1. If the user says "Yes", "No", "Exactly", or "Correct", rephrase based on the question the AI just asked.
                2. If the user picks an option (e.g., "The first one"), combine that option's details with the original problem.
                3. The output MUST be a standalone query that describes the full legal situation.
                
                History:
                {history_summary}
                
                Latest User Input: {request.query}
                
                Standalone Legal Query:"""
                
                # Optimized for query condensation
                condensed = generator.llm_service.model.generate_content(condense_prompt)
                processed_query = condensed.text.strip().replace('"', '')
                print(f"Condensed Query: {processed_query}")
            except Exception as e:
                print(f"Condense Error: {e}")
                processed_query = request.query

        # 2. Check for Statutory Anchors
        rule_context = rule_engine.find_rules(processed_query)

        # 3. Retrieve
        hits = retriever.retrieve(processed_query)
        
        # 4. Generate
        answer, enriched_sources = generator.generate_response(
            request.query, 
            hits, 
            request.history, 
            context=request.context,
            rule_context=rule_context
        )
        
        return ChatResponse(
            answer=answer,
            sources=enriched_sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
