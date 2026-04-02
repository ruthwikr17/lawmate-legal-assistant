# backend/llm/generator.py

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from backend.llm.llm_utils import LLMService

class LegalGenerator:
    def __init__(self):
        self.llm_service = LLMService()
        self.system_instruction = """
        You are LawMate, an expert legal advisor whose mission is to make Indian Law completely accessible to the everyday citizen. Your job is to provide aggressive, highly actionable legal guidance based strictly on the retrieved context, but written in incredibly simple, plain English.

        CORE PERSONA & TONE:
        - The Plain-English Translator: Write at an 10th-grade reading level. If a teenager cannot understand your sentence, rewrite it.
        - The Confident Guide: Speak with decisive authority, but explain things like you are talking to a stressed friend.
        - Power Dynamics in Simple Terms: Focus on exactly what the opposing party CANNOT do, what they must prove, and how the user is protected, using everyday words.
        - NEVER sound academic. Focus on the tactical "HOW" (the actual steps) over the legal "WHY" (the theory).

        [V67] NATURAL INTELLIGENCE PROTOCOLS (STRICT ENFORCEMENT):
        
        1. BANNED LABELS:
           - NEVER use the headers "The Bottom Line", "The Legal Basis", or "Your Action Plan." These are banned as they sound rigid and academic.
           - INSTEAD: Use simple, descriptive headers based on the content. Examples: `### Filing your written statement`, `### Getting your FIR copy`, `### What the rule says`.

        2. HYBRID AMBIGUITY (Clarification):
           - If a query is too vague (e.g., "How do I file a police complaint?"), DO NOT just ask a question.
           - FIRST: Provide a 2-3 sentence high-level overview of the most common path.
           - SECOND: Ask a single, specific clarifying question to narrow down the situation (e.g., "Is this for a theft, an accident, or harassment?").

        3. MANDATORY LEGAL CITATION & TACTICAL FOCUS:
           - YOU MUST EXPLICITLY CITE the exact Act name and Section number (e.g., **The Maharashtra Rent Control Act, 1999, Section 11**) early in your response to establish unquestionable legal authority.
           - After citing the law, pivot immediately to the "How-To" details: paper trails, Registered Post, specific forms, and physical steps.
           - Explain how the cited law protects them in 1 simple sentence without lecturing.

        4. CONDITIONAL CASE-LAW (Default = Statutes Only):
           - By default, ONLY use the Statutory Rule/Act for your answer. Ignore [Source] items marked as 'case_law' in the metadata.
           - ONLY use Case-Based Reasoning if explicitly requested or if the statute is too broad.

        STRICT VOCABULARY RULES (BAN LEGAL JARGON):
        - NEVER use dense legal terms like "indefensible," "contemporaneous," "admissible," "void," "fatal flaw," or "judicial precedents.", and other complex words.
        - Use everyday equivalents:
            - Instead of "contemporaneous Seizure Memo," say "an official receipt on the spot (Seizure Memo)."
            - Instead of "legally indefensible," say "completely against the law."
            - Instead of "statutory requirement," say "the strict rule."

        DYNAMIC NARRATIVE FLOW:
        Open immediately with the direct answer. No fluff. Use `###` headers for each distinct part of the guidance. 
        """

    def _build_context_text(self, retrieved_results: List[Dict]) -> str:
        if not retrieved_results:
            return "No specific statutory context found. Rely on general Indian legal principles."
            
        context_blocks = []
        for i, res in enumerate(retrieved_results, start=1):
            metadata = res.get("metadata", {})
            block = f"""
            [Source {i}]
            Act: {metadata.get('document_title', 'Indian Statute')}
            Section: {metadata.get('section_title', 'N/A')}
            Content: {res['text']}
            """
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

    def generate_response(self, query: str, retrieved_results: List[Dict], history: List[Dict] = None, context: Dict = None, rule_context: Optional[List[Dict]] = None) -> Tuple[str, List[Dict]]:
        """Generates a legal guidance response using the LLM with RAG and Rule Engine support."""
        context_text = self._build_context_text(retrieved_results)
        
        # 1. Process Statutory Anchors
        anchor_str = ""
        if rule_context:
            anchor_items = []
            for r in rule_context:
                anchor_items.append(f"- VERIFIED STATUTORY RULE: {r['rule']}\n  Description: {r['description']}")
            anchor_str = "\n".join(anchor_items)
            anchor_str = f"\nVERIFIED STATUTORY ANCHORS (USE THESE AS THE PRIMARY LEGAL BASIS):\n{anchor_str}\n"

        # 2. Extract User Context
        persona = context.get('persona', 'Citizen') if context else 'Citizen'
        style = context.get('style', 'Simple') if context else 'Simple'
        jurisdiction = context.get('jurisdiction', 'India') if context else 'India'
        city = context.get('city', 'Not Specified') if context else 'Not Specified'

        history_text = ""
        if history:
            for turn in history[-5:]:
                role = turn.get('role', 'user').upper()
                content = turn.get('content', '')
                history_text += f"{role}: {content}\n"

        is_dossier = "ARCHITECT DOSSIER" in query.upper() or "CREATE DOSSIER" in query.upper()

        current_system_instruction = self.system_instruction

        if is_dossier:
            dossier_directive = """
            [DOSSIER ARCHITECT MODE]
            You are building a 3-phase procedural manifest.
            1. Use PLAIN LANGUAGE to guide the user through each step.
            2. In the 'logic' field, provide 3-4 HELPFUL SENTENCES explaining the legal process clearly.
            3. Ensure the 'objective' and 'checklist' are actionable.
            4. RETURN ONLY THE JSON OBJECT.
            """
            query = f"{query}\n\n{dossier_directive}"

        prompt = f"""
        {current_system_instruction}

        [CRITICAL LEGAL MAXIMS TO FOLLOW]
        1. UNDER INDIAN LAW, A WRITTEN LEASE IS NOT MANDATORY FOR TENANCY TO EXIST. Oral agreements, possession, and rent receipts constitute a valid tenancy. NEVER tell a user they are unprotected because they lack a written lease.
        2. NEVER arbitrarily cite the "Model Tenancy Act". It is merely a template. You MUST cite the specific, enforceable Act governing the user's state.

        [CRITICAL JSON OUTPUT REGULATION]
        You MUST return your entire response as a raw JSON object ONLY. NO markdown formatting. NO ```json wrappers.
        Based on the provided Statutory Knowledge Base AND your general knowledge of Indian Law, extract the EXACT name of the Act and the actual Section you used to form your advice. 
        If there are multiple sections, list them all in the laws_cited array.
        
        REQUIRED JSON SCHEMA:
        {{
            "answer": "Your plain-English guide/advice responding to the user based on the context...",
            "laws_cited": [
                {{
                    "act": "Name of the Act (e.g., Transfer of Property Act, 1882)",
                    "section": "Section number AND TITLE (e.g., Section 106 - Lease of immovable property)"
                }}
            ]
        }}

        [USER CONTEXT]
        Active Persona: {persona}
        Preferred Style: {style}
        User Location: {city}, {jurisdiction}
        
        [JURISDICTION MANDATE]
        You MUST tailor your advice specifically to the laws of {city}, {jurisdiction}. 
        If the RAG context restricts you or lacks the exact local law, rely on your GENERAL LEGAL KNOWLEDGE of Indian statutes applicable strictly to {city}, {jurisdiction} (e.g., for Hyderabad, you MUST apply the Telangana Buildings (Lease, Rent and Eviction) Control Act, 1960).
        NEVER vaguely cite "The Rent Control Act". You must explicitly cite the exact STATE-SPECIFIC Act.
        
        {anchor_str}
        
        [STATUTORY KNOWLEDGE BASE (RAG)]
        {context_text}
        
        [CONVERSATION HISTORY]
        {history_text}
        
        USER QUERY: {query}
        
        Response:"""

        try:
            # Temperature extremely low for precise JSON formatting
            response = self.llm_service.model.generate_content(
                prompt,
                generation_config={"temperature": 0.1}
            )
            
            if not response:
                return "The AI engine returned no response.", []

            raw_text = response.text.strip()
            
            # Clean possible markdown formatting
            if raw_text.startswith("```json"):
                raw_text = raw_text.split("```json", 1)[1]
            if raw_text.endswith("```"):
                raw_text = raw_text.rsplit("```", 1)[0]
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```", 1)[1]
            raw_text = raw_text.strip()
            
            try:
                import json
                data = json.loads(raw_text)
                answer = data.get("answer", raw_text)
                ai_citations = data.get("laws_cited", [])
                
                enriched_sources = []
                seen = set()
                for cite in ai_citations:
                    act_name = cite.get("act", "Governing Statute").strip()
                    section_num = cite.get("section", "N/A").strip()
                    key = f"{act_name}-{section_num}"
                    if key not in seen and act_name:
                        seen.add(key)
                        enriched_sources.append({
                            "title": act_name,
                            "section": section_num
                        })
                
                return answer, enriched_sources
                
            except Exception as parse_error:
                print(f"JSON Parse Error: {parse_error} - Raw text: {raw_text}")
                return raw_text, [{"title": "Indian Legal Framework", "section": "Multiple Provisions"}]

        except Exception as e:
            print(f"CRITICAL ERROR IN GENERATOR: {e}")
            traceback.print_exc()
            return "LawMate encountered an internal logic error. Please try again.", []

    def analyze_notice_risk(self, extracted_text: str) -> str:
        """Analyzes a legal notice and generates a tactical Risk Manifesto."""
        
        prompt = f"""
        {self.system_instruction}
        
        [DOCUMENT ANALYSIS MODE]
        You are analyzing a legal notice or document extracted via OCR. 
        Your goal is to build a "Risk Manifesto" for the user.
        
        STRICT RULES:
        1. Identify ANY deadlines (e.g., "within 15 days", "by 5th Oct").
        2. Identify ANY financial or legal threats (e.g., "forfeit deposit", "file a suit").
        3. Identify "Paper-Trail Gaps" (e.g., if the notice claims they sent a warning before and the user doesn't have it).
        4. Give 3-4 highly tactical steps to respond or protect themselves.
        
        EXTRACTED TEXT:
        {extracted_text}
        
        Response:"""

        try:
            response = self.llm_service.model.generate_content(
                prompt,
                generation_config={"temperature": 0.2}
            )
            
            if not response:
                return "The AI engine was unable to analyze this document."

            return response.text.strip()

        except Exception as e:
            print(f"RISK ANALYSIS ERROR: {e}")
            return "LawMate encountered an error while analyzing your document. Please ensure the image is clear."
