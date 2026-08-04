import os
import uvicorn
from typing import List, Optional
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_engine import process_rag_query
from router import classify_intent

app = FastAPI(
    title="Groww Facts-Only SBI Mutual Fund AI Assistant API",
    description="Zero-hallucination, facts-only RAG backend API powered by SQLite and ChromaDB.",
    version="1.0.0"
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Request / Response Schemas
class ChatRequest(BaseModel):
    message: str = Field(..., description="User prompt or question", example="What is the minimum SIP for SBI Small Cap Direct Growth?")
    session_id: Optional[str] = Field(None, description="Optional session UUID for conversation tracking")


class ChatResponse(BaseModel):
    status: str = "success"
    intent: str
    answer: str
    source_url: str
    last_updated: str = "2026-08-04"
    follow_up_questions: List[str] = []
    clarification_needed: Optional[str] = None


@app.get("/api/health")
async def health_check():
    """Health check endpoint for service uptime monitoring."""
    return {
        "status": "online",
        "version": "1.0.0",
        "service": "Groww SBI Mutual Fund RAG Assistant"
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main conversation endpoint handling user prompts through Phase 3 RAG Engine."""
    user_msg = request.message
    if not user_msg or not user_msg.strip():
        raise HTTPException(status_code=400, detail="Message prompt cannot be empty.")

    # Step 1: Detect intent via Phase 2 router
    intent_data = classify_intent(user_msg)
    intent = intent_data["intent"]

    # Step 2: Execute query via Phase 3 RAG Engine
    rag_res = process_rag_query(user_msg)

    # Step 3: Handle special cases (Ambiguous query clarification)
    clarification = None
    if intent == "AMBIGUOUS_QUERY":
        clarification = intent_data.get("clarification_needed", "Please specify the scheme name and plan.")

    return ChatResponse(
        status="success",
        intent=intent,
        answer=rag_res["answer"],
        source_url=rag_res["source_url"],
        last_updated="2026-08-04",
        follow_up_questions=rag_res.get("follow_up_questions", []),
        clarification_needed=clarification
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
