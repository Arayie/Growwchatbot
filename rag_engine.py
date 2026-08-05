import os
import re
import logging
import concurrent.futures
from typing import Dict, Any, List, Optional

from db import get_all_scheme_facts, get_scheme_fact_by_id, DB_FILE
from router import classify_intent
from guardrails import sanitize_user_input

# Configure Structured Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("rag_engine")

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DEFAULT_TIMESTAMP = "2026-08-04"
DEFAULT_SOURCE_URL = "https://www.sbimf.com"
MAX_SIMILARITY_SCORE_THRESHOLD = 0.85

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


def _match_keyword(kw: str, text: str) -> bool:
    """Matches keyword using word boundaries for short terms like 'ter', 'nav', 'aum'."""
    if len(kw) <= 4:
        return bool(re.search(r'\b' + re.escape(kw) + r'\b', text))
    return kw in text


def _get_embedding_function():
    """
    Lightweight embedding provider selector.
    Prioritizes lightweight API embeddings (Google Gemini / OpenAI) over heavy local PyTorch models.
    """
    # 1. Check for Gemini API key
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            logger.info("Using API Embeddings: GoogleGenerativeAIEmbeddings")
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_key)
        except Exception as e:
            logger.warning(f"Could not initialize GoogleGenerativeAIEmbeddings: {e}")

    # 2. Check for OpenAI API key
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            from langchain_openai import OpenAIEmbeddings
            logger.info("Using API Embeddings: OpenAIEmbeddings")
            return OpenAIEmbeddings(openai_api_key=openai_key)
        except Exception as e:
            logger.warning(f"Could not initialize OpenAIEmbeddings: {e}")

    # 3. Fallback to local HuggingFace embeddings if available
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.info("Fallback to local HuggingFaceEmbeddings")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"local_files_only": True}
        )
    except Exception as e:
        logger.warning(f"Local HuggingFaceEmbeddings unavailable (heavy PyTorch stripped): {e}")
        return None


def synthesize_llm_answer(retrieved_context: str, user_query: str) -> str:
    """
    Executes actual LLM completion call (Groq / Gemini / OpenAI) with prompt template.
    Returns LLM generated response or LLM Generation Failed error if completion fails.
    """
    prompt = f"""You are a factual assistant for SBI Mutual Funds.
Answer the question in 2 clear, natural sentences using ONLY the facts below.
If the answer isn't in the facts, say "Information not available in official SBI AMC documents."

Facts:
{retrieved_context}

User Question: {user_query}"""

    errors = []

    # 1. Groq API (llama-3.1-8b-instant)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            import groq
            client = groq.Groq(api_key=groq_key)
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a factual assistant for SBI Mutual Funds."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.2,
                timeout=10.0
            )
            if res.choices and res.choices[0].message.content:
                logger.info("Successfully completed LLM synthesis using Groq (llama-3.1-8b-instant)")
                return res.choices[0].message.content.strip()
        except Exception as e:
            err_msg = f"Groq Error: {type(e).__name__} - {str(e)}"
            logger.warning(err_msg)
            errors.append(err_msg)

    # 2. Gemini API (gemini-2.5-flash)
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt, request_options={"timeout": 10})
            if response and response.text:
                logger.info("Successfully completed LLM synthesis using Gemini Flash (gemini-2.5-flash)")
                return response.text.strip()
        except Exception as e:
            err_msg = f"Gemini Error: {type(e).__name__} - {str(e)}"
            logger.warning(err_msg)
            errors.append(err_msg)

    # 3. OpenAI API (gpt-4o-mini)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a factual assistant for SBI Mutual Funds."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.2,
                timeout=10.0
            )
            if res.choices and res.choices[0].message.content:
                logger.info("Successfully completed LLM synthesis using OpenAI (gpt-4o-mini)")
                return res.choices[0].message.content.strip()
        except Exception as e:
            err_msg = f"OpenAI Error: {type(e).__name__} - {str(e)}"
            logger.warning(err_msg)
            errors.append(err_msg)

    # If API completion failed or keys are missing, return explicit error string
    if errors:
        return f"LLM Generation Failed: {'; '.join(errors)}"
    else:
        return "LLM Generation Failed: Missing API Key - None of GROQ_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY are configured in environment."


def query_sql_facts(query_text: str) -> Optional[Dict[str, Any]]:
    """Attempts to match scheme and metric in SQLite database."""
    query_lower = query_text.lower()

    # Identify Scheme ID
    matched_scheme_id = None
    for scheme_id, keywords in SCHEME_KEYWORDS.items():
        if any(_match_keyword(kw, query_lower) for kw in keywords):
            matched_scheme_id = scheme_id
            break

    if not matched_scheme_id:
        return None

    try:
        fact_record = get_scheme_fact_by_id(matched_scheme_id)
        if not fact_record:
            return None

        scheme_name = fact_record["scheme_name"]
        source_url = fact_record["source_url"]

        # Identify Metric
        matched_metric = None
        for metric_key, keywords in METRIC_KEYWORDS.items():
            if any(_match_keyword(kw, query_lower) for kw in keywords):
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

        logger.info(f"SQL Lookup Success: Scheme='{scheme_name}', Metric='{matched_metric}'")
        return {
            "text": ans,
            "source_url": source_url,
            "scheme_name": scheme_name,
            "last_updated": fact_record["last_updated"]
        }
    except Exception as e:
        logger.error(f"Error executing SQL fact query: {e}", exc_info=True)
        return None


def query_vector_search(query_text: str) -> Optional[Dict[str, Any]]:
    """Performs semantic similarity search in ChromaDB with lightweight API or native Chroma search."""
    if not os.path.exists(CHROMA_DB_DIR):
        logger.warning(f"ChromaDB directory not found at {CHROMA_DB_DIR}")
        return None

    try:
        # Option A: LangChain Chroma with API / local embeddings
        embeddings = _get_embedding_function()
        if embeddings:
            try:
                from langchain_community.vectorstores import Chroma
                vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
                results = vectorstore.similarity_search_with_score(query_text, k=4)
                if results:
                    best_doc, score = results[0]
                    logger.info(f"ChromaDB Vector Search Result: Best Score = {score:.4f}")
                    if score <= MAX_SIMILARITY_SCORE_THRESHOLD:
                        content = best_doc.page_content.strip()
                        metadata = best_doc.metadata
                        return {
                            "text": content,
                            "source_url": metadata.get("source_url", DEFAULT_SOURCE_URL),
                            "scheme_name": metadata.get("topic", "SBI Mutual Fund"),
                            "last_updated": metadata.get("last_updated", DEFAULT_TIMESTAMP)
                        }
            except Exception as e_lc:
                logger.warning(f"LangChain vectorstore query skipped: {e_lc}")

        # Option B: Native ChromaDB PersistentClient search (Lightweight, No PyTorch)
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collections = client.list_collections()
        if collections:
            collection = client.get_collection(collections[0].name)
            res = collection.query(query_texts=[query_text], n_results=1)
            if res and res.get("documents") and res["documents"][0]:
                doc_text = res["documents"][0][0]
                meta = res["metadatas"][0][0] if res.get("metadatas") else {}
                logger.info("ChromaDB Native Client Search Succeeded")
                return {
                    "text": doc_text,
                    "source_url": meta.get("source_url", DEFAULT_SOURCE_URL),
                    "scheme_name": meta.get("topic", "SBI Mutual Fund"),
                    "last_updated": meta.get("last_updated", DEFAULT_TIMESTAMP)
                }

        return None
    except Exception as e:
        logger.error(f"Error querying ChromaDB vector store: {e}", exc_info=True)
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
    logger.info(f"Processing RAG Query: '{user_query}'")
    
    try:
        # Step 1: Intent Classification & Safety Guardrails
        intent_data = classify_intent(user_query)
        intent = intent_data["intent"]

        # Refusal Responses for NON_MF_QUERY, ADVICE_OR_OPINION, AMBIGUOUS_QUERY
        if intent == "NON_MF_QUERY":
            logger.info("Query classified as NON_MF_QUERY")
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
            logger.info("Query classified as ADVICE_OR_OPINION")
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
            logger.info("Query classified as AMBIGUOUS_QUERY")
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

        if not retrieved:
            retrieved = query_vector_search(sanitized_query)

        # Step 3: Zero Hallucination Rule (Unknown Refusal)
        if not retrieved:
            logger.info("No matching factual source found. Returning zero-hallucination refusal.")
            unknown_msg = "Information not available in official SBI AMC documents."
            return {
                "answer": unknown_msg,
                "source_url": DEFAULT_SOURCE_URL,
                "follow_up_questions": [
                    "What is the minimum SIP amount for SBI Small Cap Fund?",
                    "How do I download my capital gains statement?",
                    "What is the lock-in period for SBI ELSS Fund?"
                ]
            }

        # Step 4: ACTUALLY call the LLM completion (do not return retrieved context directly!)
        try:
            final_answer = synthesize_llm_answer(retrieved['text'], user_query)
        except Exception as e:
            final_answer = f"LLM Generation Failed: {type(e).__name__} - {str(e)}"

        # Format Final Answer (LLM Synthesized Answer + Citation + Timestamp)
        formatted_answer = (
            f"{final_answer}\n"
            f"Source: {retrieved['source_url']}\n\n"
            f"Last updated from sources: {retrieved['last_updated']}"
        )

        follow_ups = generate_follow_up_questions(retrieved.get("scheme_name", ""))

        return {
            "answer": formatted_answer,
            "source_url": retrieved["source_url"],
            "follow_up_questions": follow_ups
        }

    except Exception as e:
        logger.error(f"Unhandled exception in process_rag_query: {e}", exc_info=True)
        fallback_err = (
            f"LLM Generation Failed: {type(e).__name__} - {str(e)}"
        )
        return {
            "answer": fallback_err,
            "source_url": DEFAULT_SOURCE_URL,
            "follow_up_questions": [
                "What is the minimum SIP amount for SBI Small Cap Fund?",
                "How do I download my capital gains statement?"
            ]
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
