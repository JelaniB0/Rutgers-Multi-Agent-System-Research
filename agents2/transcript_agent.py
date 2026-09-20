"""
Transcript agent that parses transcripts. 
"""

import json
import pdfplumber
from typing import Dict, List, Optional
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIResponsesClient
from typing import Dict, List, Optional
import re

from .shared_types import AgentResponse, ConversationState

class TranscriptAgent(ChatAgent):
    """
    Agent to parse student transcripts and extract relevant course information from Rutgers transcripts. 
    """

    SYSTEM_PROMPT = """
        You are a transcript parser for Rutgers University.
        Extract academic data from a transcript and return ONLY valid JSON — no markdown, no explanation.
    
        Course code format: "SCH:DEPT:NUM" — e.g. "01:198:111"
    
        Return exactly this structure:
        {
            "student_name": "FIRST LAST",
            "student_id": "123456789",
            "cumulative_gpa": 3.74,
            "total_degree_credits": 91.0,
            "year_standing": "Freshman|Sophomore|Junior|Senior|Graduate|Unknown",
            "academic_level": null,
            "graduate_program": null,
            "completed_courses": [
                {
                    "code": "01:198:112",
                    "title": "Data Structures",
                    "credits": 4.0,
                    "grade": "A",
                    "semester": "Fall 2024"
                }
            ],
            "in_progress_courses": [
                {
                    "code": "01:198:461",
                    "title": "Machine Learning Principles",
                    "credits": 4.0,
                    "semester": "Spring 2026"
                }
            ],
            "ap_credits": [
                {
                    "code": "01:119:115",
                    "title": "Biology",
                    "credits": 4.0
                }
            ],
            "transfer_courses": [
                {
                    "code": "01:640:152",
                    "title": "Unified Calculus II",
                    "credits": 4.0
                }
            ]
        }
    
        Rules:
        - cumulative_gpa: use the LAST cumulative avg listed (most recent)
        - total_degree_credits: use the LAST degree credits earned value
        - completed_courses: only courses with a letter grade (A, B+, etc.) or PA/P
        - in_progress_courses: current semester courses with no grade yet
        - academic_level: undergraduate or graduate only from explicit enrollment/program information; otherwise null.
        - graduate_program: masters, phd, or msds only when explicitly identified; otherwise null.
        - Do not infer current graduate status from a graduate course or an old undergraduate degree.
        - year_standing: Graduate for a graduate program. Only for confirmed undergraduate programs,
          infer from credits (<30 Freshman, <60 Sophomore, <90 Junior, 90+ Senior); otherwise Unknown.
        - ignore 0-credit duplicate lab lines
        - ignore stray single letters or watermark artifacts in the text
        """
    
    # initialize with system prompt and model. 
    def __init__(self, client: OpenAIResponsesClient, model: str):
        super().__init__(
            chat_client=client, 
            default_options={"model_id": model},
            instructions=self.SYSTEM_PROMPT
        )
        self.model = model
        # print(f"[TranscriptAgent] Initialized with model: {model}")

    async def parse_transcript(self, pdf_path: str, state: ConversationState) -> AgentResponse:
        """
        Extract text from pdf, use LLM to parse structure of PDF. 
        Stores result in state.transcript_data and returns AgentResponse.
        """

        try:
            raw_text = self._extract_text(pdf_path)

            if not raw_text.strip():
                return AgentResponse(
                    success=False,
                    errors=["Extracted text is empty. Check PDF content and extraction method."]
                )
            
            # print(f"[TranscriptAgent] Extracted {len(raw_text)} chars from PDF, parsing...")

            response = await self.run(raw_text)
            response_text = response.messages[-1].contents[0].text

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                return AgentResponse(
                    success=False,
                    errors=["LLM response does not contain valid JSON. Response: " + response_text]
                )
            
            data = json.loads(json_match.group())

            # store transcript info in conversation state for other agents to access transcript data. 
            state.transcript_data = data

            # print(f"[Transcript Agent] Parsed successfully: "
            #       f"{data.get('student_name')}, GPA: {data.get('cumulative_gpa')}, "
            #       f"Year: {data.get('year_standing')}, "
            #       f"Completed: {len(data.get('completed_courses', []))} courses")
            
            return AgentResponse(
                success=True,
                data=data,
                metadata={
                    "model_used": self.model,
                    "pdf_path": pdf_path
                }
            )

        except json.JSONDecodeError as e:
            return AgentResponse(
                success=False,
                errors=[f"Failed to parse transcript JSON: {e}"]
            )
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return AgentResponse(
                success=False,
                errors=[f"Transcript parsing error: {str(e)}"]
            )

    @staticmethod   
    def summarize_for_prompt(data: Dict) -> str:
        """
        Compact string for injecting transcript context into agent prompts
        (orchestrator, planning agent, data agent filtering).
        """

        # transcript take away of completed cs courses and in progress cs courses, and courses in general (may be irrelevant for planning but could be useful for data agent filtering)
        completed_cs = [
            c for c in data.get("completed_courses", []) if ":198:" in c.get("code", "")
        ]
        in_progress_cs = [
            c for c in data.get("in_progress_courses", []) if ":198:" in c.get("code", "")
        ]
        all_completed_str = ", ".join(
            f"{c['code']}({c.get('grade', 'P')})"
            for c in data.get("completed_courses", [])
        )

        # returns compact summary of transcript data for prompt injection.
        return (
            f"STUDENT TRANSCRIPT:\n"
            f"  Name:            {data.get('student_name')}\n"
            f"  Year Standing:   {data.get('year_standing')}\n"
            f"  Cumulative GPA:  {data.get('cumulative_gpa')}\n"
            f"  Degree Credits:  {data.get('total_degree_credits')}\n"
            f"  CS Completed:    {', '.join(c['code'] for c in completed_cs) or 'None'}\n"
            f"  CS In-Progress:  {', '.join(c['code'] for c in in_progress_cs) or 'None'}\n"
            f"  All Completed:   {all_completed_str}\n"
        )
    
    def get_completed_codes(self, data: Dict) -> set:
        """Course codes the student has already completed."""
        return {c["code"] for c in data.get("completed_courses", [])}
 
    def get_in_progress_codes(self, data: Dict) -> set:
        """Course codes the student is currently enrolled in."""
        return {c["code"] for c in data.get("in_progress_courses", [])}
    
    # assist with parsing Rutgers transcripts which have a two-column format so LLM can read since info is only passed as text. 
    # regular chat LLMs (through browsers) can read PDFS directly but, for this framework, we do not have that function. 
    
    def _extract_text(self, pdf_path: str) -> str:
        """
        Splits each page into left/right columns before extracting text.
        Rutgers transcripts use a two-column layout — without this,
        pdfplumber interleaves both columns into garbled output.
        """
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    w, h = page.width, page.height
                    for col in [page.crop((0, 0, w/2, h)), page.crop((w/2, 0, w, h))]:
                        col_text = col.extract_text()
                        if col_text:
                            text += col_text + "\n"
        except Exception as e:
            raise ValueError(f"Failed to read PDF '{pdf_path}': {e}")
        return text
 











