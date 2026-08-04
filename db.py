import sqlite3
import os
from typing import Dict, Any, List, Optional

DB_FILE = os.path.join(os.path.dirname(__file__), "sbi_scheme_facts.db")


def get_db_connection(db_path: str = DB_FILE) -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_FILE) -> None:
    """Initializes the sbi_scheme_facts table if it does not exist."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sbi_scheme_facts (
            scheme_id TEXT PRIMARY KEY,
            scheme_name TEXT NOT NULL,
            category TEXT NOT NULL,
            nav_value REAL NOT NULL,
            min_sip_amount REAL NOT NULL,
            min_lumpsum_amount REAL NOT NULL,
            fund_size_crores REAL NOT NULL,
            expense_ratio_pct REAL NOT NULL,
            exit_load_text TEXT NOT NULL,
            riskometer_rating TEXT NOT NULL,
            benchmark_index TEXT NOT NULL,
            source_url TEXT NOT NULL,
            last_updated DATE NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def upsert_scheme_fact(fact: Dict[str, Any], db_path: str = DB_FILE) -> None:
    """Inserts or updates a scheme fact record in sbi_scheme_facts."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sbi_scheme_facts (
            scheme_id, scheme_name, category, nav_value, min_sip_amount,
            min_lumpsum_amount, fund_size_crores, expense_ratio_pct,
            exit_load_text, riskometer_rating, benchmark_index, source_url, last_updated
        ) VALUES (
            :scheme_id, :scheme_name, :category, :nav_value, :min_sip_amount,
            :min_lumpsum_amount, :fund_size_crores, :expense_ratio_pct,
            :exit_load_text, :riskometer_rating, :benchmark_index, :source_url, :last_updated
        ) ON CONFLICT(scheme_id) DO UPDATE SET
            scheme_name=excluded.scheme_name,
            category=excluded.category,
            nav_value=excluded.nav_value,
            min_sip_amount=excluded.min_sip_amount,
            min_lumpsum_amount=excluded.min_lumpsum_amount,
            fund_size_crores=excluded.fund_size_crores,
            expense_ratio_pct=excluded.expense_ratio_pct,
            exit_load_text=excluded.exit_load_text,
            riskometer_rating=excluded.riskometer_rating,
            benchmark_index=excluded.benchmark_index,
            source_url=excluded.source_url,
            last_updated=excluded.last_updated
    """, fact)
    conn.commit()
    conn.close()


def get_all_scheme_facts(db_path: str = DB_FILE) -> List[Dict[str, Any]]:
    """Fetches all records from sbi_scheme_facts as a list of dicts."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sbi_scheme_facts ORDER BY scheme_name ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_scheme_fact_by_id(scheme_id: str, db_path: str = DB_FILE) -> Optional[Dict[str, Any]]:
    """Fetches a single scheme record by scheme_id."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sbi_scheme_facts WHERE scheme_id = ?", (scheme_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


if __name__ == "__main__":
    init_db()
    print("SQLite database initialized successfully at:", DB_FILE)
