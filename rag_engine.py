import os
import re
from typing import Dict, Any, List, Optional
from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from db import get_all_scheme_facts, get_scheme_fact_by_id, DB_FILE
from router import classify_intent
from guardrails import sanitize_user_input

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TIMESTAMP = "2026-08-04"
DEFAULT_SOURCE_URL = "https://www.sbimf.com"

# Scheme Mapping Keywords
SCHEME_KEYWORDS = {
    "sbi_small_cap_direct": ["small cap", "smallcap"],
    "sbi_bluechip_direct": ["bluechip", "large cap", "largecap"],
    "sbi_long_term_equity_direct": ["long term equity", "elss", "tax saver"],
    "sbi_nifty_index_direct": ["nifty index", "nifty 50"],
    "sbi_nifty_next_50_direct": ["nifty next 50", "next 50"],
    "sbi_equity_hybrid_direct": ["equity hybrid", "aggressive hybrid", "hybrid fund"],
    "sbi_gold_fund_direct": ["gold fund", "gold etf"],
    "sbi_magnum_midcap_direct": ["magnum midcap", "mid cap", "midcap"],
    "sbi_focused_equity_direct": ["focused equity", "focused fund"],
    "sbi_contra_fund_direct": ["contra fund", "contra"]
}

# Metric Mapping Keywords
METRIC_KEYWORDS = {
    "min_sip_amount": ["minimum sip", "min sip", "sip minimum", "sip amount"],
    "min_lumpsum_amount": ["minimum lump sum", "minimum lumpsum", "min lumpsum", "lump sum minimum"],
    "expense_ratio_pct": ["expense ratio", "ter", "total expense ratio"],
    "exit_load_text": ["exit load", "redemption load"],
    "nav_value": ["nav", "net asset value", "current nav"],
    "fund_size_crores": ["fund size", "aum", "asset size", "total assets"],
    "riskometer_rating": ["riskometer", "risk rating", "risk level"],
    "benchmark_index": ["benchmark", "index"]
}


def _get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"local_files_only": True}
    )
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)


def query_sql_facts(query_text: str) -> Optional[Dict[str, Any]]:
    """Attempts to match scheme and metric in SQLite database."""
    query_lower = query_text.lower()

    # Identify Scheme ID
    matched_scheme_id = None
    for scheme_id, keywords in SCHEME_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            matched_scheme_id = scheme_id
            break

    if not matched_scheme_id:
        return None

    # Retrieve Scheme Record from SQLite
    fact_record = get_scheme_fact_by_id(matched_scheme_id)
    if not fact_record:
        return None

    scheme_name = fact_record["scheme_name"]
    source_url = fact_record["source_url"]

    # Identify Metric
    matched_metric = None
    for metric_key, keywords in METRIC_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            matched_metric = metric_key
            break

    if not matched_metric:
        return None

    # Format Factual Response
    if matched_metric == "min_sip_amount":
        val = fact_record["min_sip_amount"]
        ans = f"The minimum SIP investment amount for {scheme_name} is ₹{val:g}."
    elif matched_metric == "min_lumpsum_amount":
        val = fact_record["min_lumpsum_amount"]
        ans = f"The minimum lump sum investment amount for {scheme_name} is ₹{val:g}."
    elif matched_metric == "expense_ratio_pct":
        val = fact_record["expense_ratio_pct"]
        ans = f"The Total Expense Ratio (TER) for {scheme_name} is {val:.2f}% per annum."
    elif matched_metric == "exit_load_text":
        val = fact_record["exit_load_text"]
        ans = f"The exit load structure for {scheme_name} is: {val}"
    elif matched_metric == "nav_value":
        val = fact_record["nav_value"]
        ans = f"The latest Net Asset Value (NAV) for {scheme_name} is ₹{val:.2f}."
    elif matched_metric == "fund_size_crores":
        val = fact_record["fund_size_crores"]
        ans = f"The fund size (AUM) of {scheme_name} is ₹{val:,.2f} Crores."
    elif matched_metric == "riskometer_rating":
        val = fact_record["riskometer_rating"]
        ans = f"The riskometer rating for {scheme_name} is classified as '{val}'."
    elif matched_metric == "benchmark_index":
        val = fact_record["benchmark_index"]
        ans = f"The benchmark index for {scheme_name} is the {val}."
    else:
        return None

    return {
        "text": ans,
        "source_url": source_url,
        "scheme_name": scheme_name,
        "last_updated": fact_record["last_updated"]
    }


def query_vector_search(query_text: str) -> Optional[Dict[str, Any]]:
    """Performs semantic similarity search in ChromaDB for procedural/unstructured queries."""
    if not os.path.exists(CHROMA_DB_DIR):
        return None

    try:
        vectorstore = _get_vectorstore()
        results = vectorstore.similarity_search_with_score(query_text, k=4)
        if not results:
            return None

        best_doc, score = results[0]
        # Distance score check for relevance (tight threshold for zero hallucination)
        if score > 0.85:
            return None

        content = best_doc.page_content.strip()
        # Take top 2 clean sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if len(s.strip()) > 10]
        clean_text = " ".join(sentences[:2]) if sentences else content[:250]
        if not clean_text.endswith("."):
            clean_text += "."

        metadata = best_doc.metadata
        source_url = metadata.get("source_url", DEFAULT_SOURCE_URL)
        last_updated = metadata.get("last_updated", DEFAULT_TIMESTAMP)

        return {
            "text": clean_text,
            "source_url": source_url,
            "scheme_name": metadata.get("topic", "SBI Mutual Fund"),
            "last_updated": last_updated
        }
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return None


def generate_follow_up_questions(scheme_name: str) -> List[str]:
    """Generates 3 relevant follow-up questions based on current scheme/topic context."""
    s_name = scheme_name.lower()
    if "small cap" in s_name:
        return [
            "What is the exit load for SBI Small Cap Fund Direct Growth?",
            "What is the expense ratio for this scheme?",
            "What is the current fund size (AUM) of this fund?"
        ]
    elif "bluechip" in s_name:
        return [
            "What is the minimum SIP amount for SBI Bluechip Fund?",
            "What is the benchmark index for SBI Bluechip Fund?",
            "What is the exit load if redeemed within 1 year?"
        ]
    elif "elss" in s_name or "long term equity" in s_name:
        return [
            "What is the lock-in period for SBI Long Term Equity Fund?",
            "Can I claim tax deduction under Section 80C for this fund?",
            "What is the minimum SIP for SBI ELSS Tax Saver?"
        ]
    elif "gold" in s_name:
        return [
            "What is the investment objective of SBI Gold Fund?",
            "What is the exit load for SBI Gold Fund?",
            "What is the benchmark index for SBI Gold Fund?"
        ]
    elif "nifty" in s_name:
        return [
            "What is the expense ratio of SBI Nifty Index Fund Direct Plan?",
            "What is the exit load if redeemed within 7 days?",
            "What is the benchmark index for SBI Nifty Index Fund?"
        ]
    else:
        return [
            "How do I download my capital gains statement?",
            "What is the exit load if redeemed within 1 year?",
            "What is the minimum SIP amount for SBI funds?"
        ]


def process_rag_query(user_query: str) -> Dict[str, Any]:
    """
    Unified RAG Execution Pipeline combining Guardrails (Phase 2),
    SQL Facts + Vector Search (Phase 1), and Response Formatting (Phase 3).
    """
    # Step 1: Intent Classification & Safety Guardrails
    intent_data = classify_intent(user_query)
    intent = intent_data["intent"]

    # Refusal Responses for NON_MF_QUERY, ADVICE_OR_OPINION, AMBIGUOUS_QUERY
    if intent == "NON_MF_QUERY":
        return {
            "answer": intent_data["response"],
            "source_url": DEFAULT_SOURCE_URL,
            "follow_up_questions": [
                "What is the minimum SIP for SBI Small Cap Fund?",
                "How do I download my account statement?",
                "What is the NAV cut-off time for equity funds?"
            ]
        }
    elif intent == "ADVICE_OR_OPINION":
        refusal_msg = f"{intent_data['response']}\n\nOfficial Factsheets: {intent_data['citation_url']}\n\nLast updated from sources: {DEFAULT_TIMESTAMP}"
        return {
            "answer": refusal_msg,
            "source_url": intent_data["citation_url"],
            "follow_up_questions": [
                "What is the riskometer rating of SBI Small Cap Fund?",
                "What is the expense ratio of SBI Bluechip Fund?",
                "What is the benchmark for SBI Long Term Equity Fund?"
            ]
        }
    elif intent == "AMBIGUOUS_QUERY":
        return {
            "answer": intent_data["clarification_needed"],
            "source_url": DEFAULT_SOURCE_URL,
            "follow_up_questions": [
                "What is the NAV of SBI Small Cap Direct Growth?",
                "What is the NAV of SBI Bluechip Direct Growth?",
                "What is the NAV of SBI Long Term Equity Direct Growth?"
            ]
        }

    # Step 2: Hybrid Retrieval (SQL Lookup -> Vector Search Fallback)
    sanitized_query = intent_data.get("sanitized_query", user_query)
    route = intent_data.get("route", "SQL_LOOKUP")

    retrieved = None
    if route == "SQL_LOOKUP":
        retrieved = query_sql_facts(sanitized_query)

    # Fallback to Vector Search if SQL Lookup returned None or for VECTOR_SEARCH route
    if not retrieved:
        retrieved = query_vector_search(sanitized_query)

    # Step 3: Zero Hallucination Rule (Unknown Refusal)
    if not retrieved:
        unknown_msg = (
            "I do not have enough verified factual information in my current sources to answer this question. "
            "Please refer to the official SBI Mutual Fund documentation at https://www.sbimf.com."
        )
        return {
            "answer": unknown_msg,
            "source_url": DEFAULT_SOURCE_URL,
            "follow_up_questions": [
                "What is the minimum SIP amount for SBI Small Cap Fund?",
                "How do I download my capital gains statement?",
                "What is the lock-in period for SBI ELSS Fund?"
            ]
        }

    # Step 4: Format Final Answer (<= 3 Sentences + Citation + Timestamp)
    answer_text = (
        f"{retrieved['text']}\n"
        f"Source: {retrieved['source_url']}\n\n"
        f"Last updated from sources: {retrieved['last_updated']}"
    )

    follow_ups = generate_follow_up_questions(retrieved.get("scheme_name", ""))

    return {
        "answer": answer_text,
        "source_url": retrieved["source_url"],
        "follow_up_questions": follow_ups
    }


if __name__ == "__main__":
    sample_queries = [
        "What is the minimum SIP for SBI Small Cap Direct Growth?",
        "How do I download my capital gains statement?",
        "Should I invest in SBI Small Cap Fund?",
        "What is the capital of France?",
        "What is the NAV?"
    ]

    for q in sample_queries:
        print("=" * 70)
        print(f"User Query: '{q}'")
        res = process_rag_query(q)
        print("Response Answer:\n" + res["answer"])
        print("Source URL:", res["source_url"])
        print("Follow-up Questions:", res["follow_up_questions"])
