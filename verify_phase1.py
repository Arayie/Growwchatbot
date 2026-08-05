import os
import pandas as pd
from db import get_all_scheme_facts, DB_FILE
from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def verify_sqlite_db():
    print("=" * 80)
    print("1. VERIFYING SQLITE DATABASE (sbi_scheme_facts)")
    print("=" * 80)
    
    if not os.path.exists(DB_FILE):
        print(f"Error: Database file {DB_FILE} does not exist!")
        return

    facts = get_all_scheme_facts()
    df = pd.DataFrame(facts)
    
    if df.empty:
        print("Warning: sbi_scheme_facts table is empty!")
    else:
        print(f"Total Scheme Facts Records in SQLite: {len(df)}\n")
        # Format key columns for clean display
        display_cols = [
            "scheme_id", "scheme_name", "category", "nav_value", 
            "min_sip_amount", "expense_ratio_pct", "riskometer_rating", "benchmark_index"
        ]
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(df[display_cols].to_string(index=False))


def verify_chroma_vectorstore():
    print("\n" + "=" * 80)
    print("2. VERIFYING CHROMADB VECTOR STORE (./chroma_db)")
    print("=" * 80)

    if not os.path.exists(CHROMA_DB_DIR):
        print(f"Error: ChromaDB directory {CHROMA_DB_DIR} does not exist!")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"local_files_only": True}
    )
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

    test_query = "What is expense ratio of SBI Contra Fund Direct Plan Growth"
    print(f"Executing Vector Search Query: '{test_query}'...\n")

    results = vectorstore.similarity_search_with_score(test_query, k=2)

    if not results:
        print("No matching vector chunks found!")
        return

    for idx, (doc, score) in enumerate(results):
        print(f"--- MATCH #{idx + 1} (Score: {score:.4f}) ---")
        print(f"Topic: {doc.metadata.get('topic')}")
        print(f"Source URL: {doc.metadata.get('source_url')}")
        print(f"Last Updated: {doc.metadata.get('last_updated')}")
        print(f"Content Snippet: {doc.page_content[:300]}...")
        print("-" * 60)


if __name__ == "__main__":
    verify_sqlite_db()
    verify_chroma_vectorstore()
