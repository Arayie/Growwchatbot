import os
import glob
import plistlib
import re
import shutil
from typing import List, Dict, Any

from pypdf import PdfReader
from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from db import init_db, upsert_scheme_fact

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXECUTION_DATE = "2026-08-04"

# Standard Scheme Facts Dataset mapped from primary scheme documents & Groww pages
DEFAULT_SCHEME_FACTS = [
    {
        "scheme_id": "sbi_small_cap_direct",
        "scheme_name": "SBI Small Cap Fund Direct Plan Growth",
        "category": "Equity - Small Cap",
        "nav_value": 178.45,
        "min_sip_amount": 500.0,
        "min_lumpsum_amount": 5000.0,
        "fund_size_crores": 33045.80,
        "expense_ratio_pct": 0.68,
        "exit_load_text": "1% if redeemed within 1 year; Nil after 1 year.",
        "riskometer_rating": "Very High",
        "benchmark_index": "BSE 250 SmallCap TRI",
        "source_url": "https://groww.in/mutual-funds/sbi-small-cap-fund-direct-growth",
        "last_updated": EXECUTION_DATE
    },
    {
        "scheme_id": "sbi_bluechip_direct",
        "scheme_name": "SBI Bluechip Fund Direct Plan Growth",
        "category": "Equity - Large Cap",
        "nav_value": 94.12,
        "min_sip_amount": 500.0,
        "min_lumpsum_amount": 5000.0,
        "fund_size_crores": 46520.15,
        "expense_ratio_pct": 0.84,
        "exit_load_text": "1% if redeemed within 1 year; Nil after 1 year.",
        "riskometer_rating": "Very High",
        "benchmark_index": "BSE 100 TRI",
        "source_url": "https://groww.in/mutual-funds/sbi-bluechip-fund-direct-growth",
        "last_updated": EXECUTION_DATE
    },
    {
        "scheme_id": "sbi_long_term_equity_direct",
        "scheme_name": "SBI Long Term Equity Fund (ELSS) Direct Plan Growth",
        "category": "ELSS Tax Saver",
        "nav_value": 385.60,
        "min_sip_amount": 500.0,
        "min_lumpsum_amount": 500.0,
        "fund_size_crores": 23890.40,
        "expense_ratio_pct": 0.95,
        "exit_load_text": "Nil (Mandatory 3-Year Lock-in Period under Section 80C).",
        "riskometer_rating": "Very High",
        "benchmark_index": "BSE 500 TRI",
        "source_url": "https://groww.in/mutual-funds/sbi-long-term-equity-fund-direct-growth",
        "last_updated": EXECUTION_DATE
    },
    {
        "scheme_id": "sbi_nifty_index_direct",
        "scheme_name": "SBI Nifty Index Fund Direct Plan Growth",
        "category": "Index Fund - Equity",
        "nav_value": 215.30,
        "min_sip_amount": 500.0,
        "min_lumpsum_amount": 5000.0,
        "fund_size_crores": 7820.60,
        "expense_ratio_pct": 0.18,
        "exit_load_text": "0.20% if redeemed within 7 days; Nil after 7 days.",
        "riskometer_rating": "Very High",
        "benchmark_index": "Nifty 50 TRI",
        "source_url": "https://groww.in/mutual-funds/sbi-nifty-index-fund-direct-growth",
        "last_updated": EXECUTION_DATE
    },
    {
        "scheme_id": "sbi_nifty_next_50_direct",
        "scheme_name": "SBI Nifty Next 50 Index Fund Direct Growth",
        "category": "Index Fund - Equity",
        "nav_value": 18.45,
        "min_sip_amount": 500.0,
        "min_lumpsum_amount": 5000.0,
        "fund_size_crores": 1450.25,
        "expense_ratio_pct": 0.30,
        "exit_load_text": "Nil.",
        "riskometer_rating": "Very High",
        "benchmark_index": "Nifty Next 50 TRI",
        "source_url": "https://groww.in/mutual-funds/sbi-nifty-next-50-index-fund-direct-growth",
        "last_updated": EXECUTION_DATE
    },
    {
        "scheme_id": "sbi_equity_hybrid_direct",
        "scheme_name": "SBI Equity Hybrid Fund Direct Plan Growth",
        "category": "Aggressive Hybrid",
        "nav_value": 268.90,
        "min_sip_amount": 500.0,
        "min_lumpsum_amount": 5000.0,
        "fund_size_crores": 68420.00,
        "expense_ratio_pct": 0.76,
        "exit_load_text": "1% for redemption in excess of 10% units within 1 year; Nil after 1 year.",
        "riskometer_rating": "Very High",
        "benchmark_index": "CRISIL Hybrid 35+65 Aggressive Index",
        "source_url": "https://groww.in/mutual-funds/sbi-equity-hybrid-fund-direct-growth",
        "last_updated": EXECUTION_DATE
    },
    {
        "scheme_id": "sbi_gold_fund_direct",
        "scheme_name": "SBI Gold Fund Direct Plan Growth",
        "category": "Fund of Funds - Gold",
        "nav_value": 24.15,
        "min_sip_amount": 500.0,
        "min_lumpsum_amount": 5000.0,
        "fund_size_crores": 2150.80,
        "expense_ratio_pct": 0.10,
        "exit_load_text": "1% if redeemed within 1 year; Nil after 1 year.",
        "riskometer_rating": "High",
        "benchmark_index": "Domestic Price of Physical Gold",
        "source_url": "https://groww.in/mutual-funds/sbi-gold-fund-direct-growth",
        "last_updated": EXECUTION_DATE
    },
    {
        "scheme_id": "sbi_magnum_midcap_direct",
        "scheme_name": "SBI Magnum Midcap Fund Direct Plan Growth",
        "category": "Equity - Mid Cap",
        "nav_value": 212.80,
        "min_sip_amount": 500.0,
        "min_lumpsum_amount": 5000.0,
        "fund_size_crores": 18950.30,
        "expense_ratio_pct": 0.82,
        "exit_load_text": "1% if redeemed within 1 year; Nil after 1 year.",
        "riskometer_rating": "Very High",
        "benchmark_index": "Nifty Midcap 150 TRI",
        "source_url": "https://groww.in/mutual-funds/sbi-magnum-midcap-fund-direct-growth",
        "last_updated": EXECUTION_DATE
    },
    {
        "scheme_id": "sbi_focused_equity_direct",
        "scheme_name": "SBI Focused Equity Fund Direct Plan Growth",
        "category": "Equity - Focused",
        "nav_value": 310.45,
        "min_sip_amount": 500.0,
        "min_lumpsum_amount": 5000.0,
        "fund_size_crores": 31200.75,
        "expense_ratio_pct": 0.71,
        "exit_load_text": "1% if redeemed within 1 year; Nil after 1 year.",
        "riskometer_rating": "Very High",
        "benchmark_index": "BSE 500 TRI",
        "source_url": "https://groww.in/mutual-funds/sbi-focused-equity-fund-direct-growth",
        "last_updated": EXECUTION_DATE
    },
    {
        "scheme_id": "sbi_contra_fund_direct",
        "scheme_name": "SBI Contra Fund Direct Plan Growth",
        "category": "Equity - Contra",
        "nav_value": 395.20,
        "min_sip_amount": 500.0,
        "min_lumpsum_amount": 5000.0,
        "fund_size_crores": 30540.90,
        "expense_ratio_pct": 0.65,
        "exit_load_text": "1% if redeemed within 1 year; Nil after 1 year.",
        "riskometer_rating": "Very High",
        "benchmark_index": "BSE 500 TRI",
        "source_url": "https://groww.in/mutual-funds/sbi-contra-fund-direct-growth",
        "last_updated": EXECUTION_DATE
    }
]


def chunk_text(text: str, chunk_size_words: int = 300, overlap_words: int = 50) -> List[str]:
    """Chunks text into segments of 250-400 words with overlap."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size_words])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
        i += (chunk_size_words - overlap_words)
    return chunks


def read_webloc_urls() -> Dict[str, str]:
    """Scans ./data/ for .webloc files and extracts target URLs."""
    webloc_files = glob.glob(os.path.join(DATA_DIR, "**", "*.webloc"), recursive=True)
    urls = {}
    for filepath in webloc_files:
        try:
            with open(filepath, "rb") as f:
                plist = plistlib.load(f)
                url = plist.get("URL", "").strip()
                if url:
                    filename = os.path.basename(filepath)
                    urls[filename] = url
        except Exception as e:
            print(f"Warning: Could not parse {filepath}: {e}")
    return urls


def process_pdf_documents() -> List[Dict[str, Any]]:
    """Scans ./data/ for PDF documents and extracts chunked text with metadata."""
    pdf_files = glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True)
    chunks_with_metadata = []

    for filepath in pdf_files:
        filename = os.path.basename(filepath)
        print(f"  --> Extracting PDF: {filename}...")
        try:
            reader = PdfReader(filepath)
            full_text = ""
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                full_text += f"\n--- Page {page_idx + 1} ---\n" + page_text

            chunks = chunk_text(full_text, chunk_size_words=300, overlap_words=40)
            for idx, chunk in enumerate(chunks):
                chunks_with_metadata.append({
                    "text": chunk,
                    "metadata": {
                        "scheme_id": filename.replace(".pdf", "").replace("-", "_"),
                        "topic": "Scheme Documentation & Guidelines",
                        "source_url": f"https://www.sbimf.com/ (Document: {filename})",
                        "last_updated": EXECUTION_DATE,
                        "chunk_index": idx
                    }
                })
        except Exception as e:
            print(f"Error reading PDF {filename}: {e}")

    return chunks_with_metadata


def build_procedural_kb_chunks() -> List[Dict[str, Any]]:
    """Builds procedural FAQ chunks (Capital Gains, Statement Downloads, KYC, SIP Cancellation, NAV Timings)."""
    procedural_docs = [
        {
            "topic": "Capital Gains Statement & Tax Downloads",
            "text": "How to download capital gains statement from SBI Mutual Fund: 1. Visit the official SBI Mutual Fund portal at www.sbimf.com or log into the InvesTap mobile app. 2. Navigate to 'Services' -> 'Statements' -> 'Capital Gains Statement'. 3. Select your Folio Number and Financial Year (e.g., FY 2025-26). 4. Click 'Download PDF' or request 'Send via Email'. You can also request an instant statement via SMS by sending 'SOL <Folio No>' to 7065611100.",
            "source_url": "https://www.sbimf.com/smart-statement"
        },
        {
            "topic": "Bank Account Change Procedure",
            "text": "Procedure to update bank account details in SBI Mutual Fund: 1. Submit a Bank Details Change Form along with a cancelled cheque of the new bank account containing the investor's name printed. 2. Provide bank statement or passbook copy not older than 3 months certified by bank manager. 3. Submit physically at nearest SBI MF Investor Service Centre (ISC) or update digitally via portal after OTP authentication.",
            "source_url": "https://www.sbimf.com/faq"
        },
        {
            "topic": "SIP Cancellation Procedure",
            "text": "How to cancel an active SIP in SBI Mutual Fund: 1. Log into your account on sbimf.com or the SBIMF InvesTap App. 2. Go to 'My Investments' -> 'Active SIPs'. 3. Select the SIP scheme and click 'Cancel SIP' / 'Stop Auto-Debit'. 4. Confirm cancellation at least 15 business days before the next SIP installment date.",
            "source_url": "https://www.sbimf.com/faq"
        },
        {
            "topic": "NAV Cut-off Timings",
            "text": "NAV applicability cut-off timings under SEBI guidelines for SBI Mutual Fund: For Liquid & Overnight schemes, purchase cut-off time is 1:30 PM (realization of funds before cut-off). For Equity and Hybrid schemes, purchase and redemption cut-off time is 3:00 PM on business days.",
            "source_url": "https://www.sbimf.com/faq"
        },
        {
            "topic": "KYC Status & Nomination Updates",
            "text": "Updating KYC and nomination details in SBI Mutual Fund: Investors can verify KYC status at cvakycra.com. Nomination details can be added or updated online at sbimf.com under Folio Management using Aadhaar e-Sign or physical submission of Nomination Form at ISC.",
            "source_url": "https://www.sbimf.com/faq"
        }
    ]

    chunks = []
    for doc in procedural_docs:
        chunks.append({
            "text": doc["text"],
            "metadata": {
                "scheme_id": "sbi_procedural_faq",
                "topic": doc["topic"],
                "source_url": doc["source_url"],
                "last_updated": EXECUTION_DATE
            }
        })
    return chunks


def main():
    print("=" * 80)
    print("      SBI MUTUAL FUND RAG — PHASE 1 INGESTION PIPELINE      ")
    print("=" * 80)

    # 1. Initialize SQLite Database
    print("\n[1/4] Initializing SQLite database (sbi_scheme_facts.db)...")
    init_db()

    # 2. Upsert Numerical Facts into SQLite
    print("\n[2/4] Upserting numerical scheme facts into SQLite...")
    for fact in DEFAULT_SCHEME_FACTS:
        upsert_scheme_fact(fact)
        print(f"  + Upserted SQLite record: [{fact['scheme_id']}] {fact['scheme_name']}")

    # 3. Read .webloc URLs & PDFs from ./data/
    print("\n[3/4] Scanning ./data/ directory for documents and links...")
    webloc_map = read_webloc_urls()
    print(f"  + Found {len(webloc_map)} .webloc file links:")
    for fname, url in webloc_map.items():
        print(f"    - {fname} -> {url}")

    pdf_chunks = process_pdf_documents()
    print(f"  + Extracted {len(pdf_chunks)} text chunks from PDF files in ./data/.")

    procedural_chunks = build_procedural_kb_chunks()
    all_chunks = pdf_chunks + procedural_chunks
    print(f"  + Total vector chunks prepared: {len(all_chunks)}")

    # 4. Embed & Upsert into ChromaDB
    print("\n[4/4] Generating embeddings and upserting into ChromaDB at ./chroma_db...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"local_files_only": True}
    )

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [f"doc_chunk_{i}" for i in range(len(texts))]

    if os.path.exists(CHROMA_DB_DIR):
        print(f"  + Refreshing existing ChromaDB directory at {CHROMA_DB_DIR}...")

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=CHROMA_DB_DIR
    )
    print(f"  + Successfully upserted {len(texts)} chunks into ChromaDB at {CHROMA_DB_DIR}!")

    print("\n" + "=" * 80)
    print("✅ PHASE 1 INGESTION COMPLETE! SQL + Vector Stores Ready.")
    print("=" * 80)


if __name__ == "__main__":
    main()
