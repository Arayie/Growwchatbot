import os
import logging
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_engine import process_rag_query
from router import classify_intent
from guardrails import sanitize_user_input

# Configure Structured Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("api_server")

app = FastAPI(
    title="Groww Facts-Only SBI Mutual Fund AI Assistant API",
    description="Zero-hallucination, facts-only RAG backend API powered by SQLite and ChromaDB.",
    version="1.0.0"
)

# Enable CORS for all origins, methods, and headers to fix preflight requests
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
    logger.info("Health check endpoint pinged.")
    return {
        "status": "online",
        "version": "1.0.0",
        "service": "Groww SBI Mutual Fund RAG Assistant"
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main conversation endpoint executing full Phase 1-5 pipeline:
    1. Guardrails (PII scrubbing via guardrails.py)
    2. Intent Router (Classification via router.py)
    3. Retrieval & LLM Synthesis (rag_engine.py)
    4. Dynamic Follow-ups Generation
    """
    user_msg = request.message
    if not user_msg or not user_msg.strip():
        logger.warning("Empty message received at /api/chat")
        raise HTTPException(status_code=400, detail="Message prompt cannot be empty.")

    logger.info(f"Received API Chat Request: '{user_msg[:60]}...'")

    try:
        # Step 1: PII Guardrail Scrubbing
        sanitized_msg = sanitize_user_input(user_msg)

        # Step 2: Intent Classification & Routing
        intent_data = classify_intent(sanitized_msg)
        intent = intent_data["intent"]

        # Step 3: Execute Retrieval, LLM Synthesis, and Dynamic Follow-ups
        rag_res = process_rag_query(sanitized_msg)

        # Step 4: Handle Special Intent Clarifications
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
    except Exception as e:
        logger.error(f"Unhandled exception in /api/chat endpoint: {e}", exc_info=True)
        fallback_msg = (
            "I do not have enough verified factual information in my current sources to answer this question. "
            "Please refer to the official SBI Mutual Fund documentation at https://www.sbimf.com."
        )
        return ChatResponse(
            status="success",
            intent="FACTUAL_QUERY",
            answer=fallback_msg,
            source_url="https://www.sbimf.com",
            last_updated="2026-08-04",
            follow_up_questions=[
                "What is the minimum SIP amount for SBI Small Cap Fund?",
                "How do I download my capital gains statement?"
            ],
            clarification_needed=None
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
