import re
from typing import Dict, Any
from guardrails import sanitize_user_input

# Advice and Recommendation Keywords
ADVICE_KEYWORDS = [
    "should i buy", "should i sell", "should i invest", "which is best", 
    "which fund is best", "which scheme is best", "is this fund good", "is this scheme good",
    "recommend", "recommendation", "suggestion", "advice", "predict", "prediction",
    "future returns", "double my money", "where should i invest", "best fund for",
    "is sbi bluechip better than", "which is safer", "should i stop my sip", "guaranteed returns"
]

# Non-Mutual Fund Keywords & Topics
NON_MF_PATTERNS = [
    r'\bcapital of\b', r'\bweather\b', r'\brecipe\b', r'\bcoding\b', r'\bpython script\b',
    r'\bwho is the president\b', r'\bprime minister\b', r'\bcricket score\b', r'\bipl match\b',
    r'\bstock pick\b', r'\bbitcoin\b', r'\bcrypto\b', r'\btell me a joke\b'
]

# Core Mutual Fund Domain Keywords
MF_DOMAIN_KEYWORDS = [
    "sbi", "fund", "scheme", "nav", "sip", "elss", "mutual", "portfolio", "sebi",
    "equity", "debt", "etf", "fof", "hybrid", "gold", "nifty", "index", "cut-off",
    "statement", "folio", "redemption", "exit load", "expense ratio", "growth",
    "dividend", "idcw", "amc", "benchmark", "holding", "asset", "lock-in", "invest",
    "riskometer", "capital gains", "ter", "aum", "fund size", "minimum sip", "lumpsum", "stp", "swp", "kyc"
]

# Specific Metric Keywords requiring SQL Lookup vs Vector Search
SQL_METRIC_KEYWORDS = ["nav", "expense ratio", "ter", "min sip", "minimum sip", "min lumpsum", "fund size", "aum", "exit load", "riskometer", "benchmark"]


def classify_intent(query: str) -> Dict[str, Any]:
    """
    Sanitizes user input and classifies incoming query into one of 4 strict intents:
    1. NON_MF_QUERY
    2. ADVICE_OR_OPINION
    3. AMBIGUOUS_QUERY
    4. FACTUAL_QUERY
    """
    if not query or not query.strip():
        return {
            "intent": "NON_MF_QUERY",
            "response": "Please enter a valid query regarding SBI Mutual Fund schemes."
        }

    # Step 1: Sanitize PII
    sanitized_query = sanitize_user_input(query.strip())
    query_lower = sanitized_query.lower()

    # Step 2: Check for Non-Mutual Fund Queries
    is_non_mf = any(re.search(pat, query_lower) for pat in NON_MF_PATTERNS)
    has_mf_keyword = any(kw in query_lower for kw in MF_DOMAIN_KEYWORDS)

    if is_non_mf or (not has_mf_keyword and not any(m in query_lower for m in SQL_METRIC_KEYWORDS)):
        return {
            "intent": "NON_MF_QUERY",
            "sanitized_query": sanitized_query,
            "response": "I am programmed to assist strictly with Mutual Fund factual queries. Please ask me about scheme specs, NAVs, SIP minimums, exit loads, or statement downloads!"
        }

    # Step 3: Check for Advice / Opinion / Recommendation Queries
    if any(kw in query_lower for kw in ADVICE_KEYWORDS):
        return {
            "intent": "ADVICE_OR_OPINION",
            "sanitized_query": sanitized_query,
            "response": "I am a facts-only assistant and cannot provide investment advice or recommendations. Please review the official scheme factsheets to make informed decisions.",
            "citation_url": "https://www.sbimf.com/factsheets"
        }

    # Step 4: Check for Ambiguous Queries (Very short / missing scheme parameters)
    # E.g. "What is the NAV?", "Tell me the exit load", "What is the expense ratio?"
    words = query_lower.split()
    is_ambiguous_short = len(words) <= 5 and any(m in query_lower for m in ["nav", "expense ratio", "exit load", "minimum sip"])
    has_scheme_name = any(s in query_lower for s in ["small cap", "bluechip", "long term equity", "elss", "nifty", "hybrid", "gold", "midcap", "focused", "contra"])

    if is_ambiguous_short and not has_scheme_name:
        return {
            "intent": "AMBIGUOUS_QUERY",
            "sanitized_query": sanitized_query,
            "clarification_needed": "Did you mean SBI Small Cap Fund - Direct Plan (Growth)? Please specify the scheme name and plan."
        }

    # Step 5: Factual Query (SQL Lookup vs Vector Search)
    is_sql_route = any(m in query_lower for m in SQL_METRIC_KEYWORDS)
    route_flag = "SQL_LOOKUP" if is_sql_route else "VECTOR_SEARCH"

    return {
        "intent": "FACTUAL_QUERY",
        "route": route_flag,
        "sanitized_query": sanitized_query
    }


if __name__ == "__main__":
    test_queries = [
        "What is the capital of France?",
        "Should I buy SBI Bluechip Fund right now?",
        "My PAN is ABCDE1234F, what is the minimum SIP for SBI Small Cap Direct Growth?",
        "What is the NAV?"
    ]
    for q in test_queries:
        res = classify_intent(q)
        print(f"Query: '{q}'\n  -> Result: {res}\n")
